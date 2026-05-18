# TracIn / IIF (Influence-based Instance Filtering) Workflow

This document traces the TracIn pipeline from launch script to final weight update, file by file and function by function.

---

## 1. Entry Point — `scripts/run_train_iif.sh`

The shell script launches training with these key flags:

```
--tracin               # activate TracIn mode
--with_validation      # use a separate held-out validation set
--val_loss_type=seqloss-lastadv
--batch_size=256       # N training samples per step
--tracin_batch_size=64 # ghost-gradient mini-batch size
--tracin_val_batch_size=64
--val_size=1024        # M validation samples
--ppo_epochs=4
--mini_batch_size=1
```

---

## 2. `scripts/train_rlhf.py` — Setup and Routing

**What it does:**
- Parses all CLI args into `ScriptArguments` via `HfArgumentParser`
- Calls `load_models(script_args)` → returns `config, tokenizer, model, optimizer, reward_model, reward_tokenizer`
- Builds the training dataset (e.g. `build_toxicity_promptdata`) and the **separate validation dataset** (e.g. `valid_dataset`)
- Creates `PPOTrainer(config, model, ref_model=None, tokenizer, dataset, data_collator, optimizer)`
  - Internally, `PPOTrainer.__init__` registers **forward and backward hooks** on every LoRA layer and on `v_head.summary` (see Section 4)
- Routes to `train_loop_with_validation(..., use_datainf=False)` when `--tracin --with_validation` is set

---

## 3. `rlhfutils/rl_utils.py` → `train_loop_with_validation` — Outer Training Loop

**Runs for `steps` iterations. Each iteration:**

### 3a. Response Generation — `get_rollouts`
- Calls `ppo_trainer.generate(question_tensors, ...)` for **training questions** → `response_tensors`
- Calls `ppo_trainer.generate(val_question_tensors, ...)` for **validation questions** → `val_response_tensors`
- Both use the current policy (no grad, `torch.no_grad()`)

### 3b. Reward Scoring — `process_reward`
- Runs the reward model (e.g. `roberta-hate-speech-dynabench-r4-target`) on formatted `(query, response)` texts
- Returns scalar reward per sample for both training and validation batches
- Rewards are converted to `torch.tensor` and sent to device

### 3c. Dispatch to `ppo_trainer.step_with_validation`
```python
stats = ppo_trainer.step_with_validation(
    question_tensors, response_tensors, rewards,
    val_question_tensors, val_response_tensors, val_rewards,
    timing, gen_data_dir=...
)
```

---

## 4. `scripts/ppo_trainer.py` — Hook Registration (done once in `__init__`)

Before any training step runs, `PPOTrainer.__init__` registers persistent hooks on every LoRA layer:

**Forward hook on `lora_A`** — fires during any forward pass when `self._record_ghost = True`:
- Captures `x` (input to lora_A, shape `[B, S, d_in]`) into `self._xs[name]`
- Captures `h = lora_A(x)` (output, shape `[B, S, r]`) into `self._hs[name]`

**Backward hook on `lora_B`** — fires during any backward pass when `self._record_ghost = True`:
- Captures `g_h` (gradient w.r.t. lora_B input, shape `[B, S, r]`) into `self._gAs[name]`
- Captures `g_o` (gradient w.r.t. lora_B output, shape `[B, S, d_out]`) into `self._gBs[name]`

**Why these four quantities?** For a LoRA layer, the full gradient matrix `∇_A L` and `∇_B L` are outer products:
```
∇_A L_i  =  g_h_i^T ⊗ x_i     (shape r × d_in)
∇_B L_i  =  g_o_i^T ⊗ h_i     (shape d_out × r)
```
Storing the four smaller factors avoids materializing the full gradient matrices.

**`ghost_mode` context manager** — temporarily replaces `optimizer.step` with a no-op so that a backward pass captures gradients into the hooks **without actually updating weights**.

---

## 5. `scripts/ppo_trainer.py` → `step_with_validation` — Per-Step Logic

### Phase 1: Global Collation

```python
model_inputs = self.prepare_model_inputs(queries, responses)
val_model_inputs = self.prepare_model_inputs(val_queries, val_responses)
```

Both the full training batch (256) and the full validation set (1024) are collated into padded tensors **once** using the **global max sequence length** across each respective set. This is critical — all chunks later will slice from these globally-padded tensors so every sub-batch has the same sequence length `S`, ensuring uniform gradient magnitudes.

If distributed: `pad_across_processes` aligns lengths across GPUs.

`model.eval()` is called and dropout is disabled to get deterministic gradients.

---

### Phase 2: Training Ghost Backward — Capture per-sample gradient factors

Loop over training mini-batches of size `tracin_batch_size=64` (4 iterations for 256 samples):

For each mini-batch:

**a) Forward pass (with hooks ON)**
```python
self._record_ghost = True
tb_logprobs, tb_logits, tb_values, tb_masks = self.batched_forward_pass(...)
self._record_ghost = False
```
- `batched_forward_pass` runs `model(**input_kwargs)` → gets `logits` and `values` from the model
- Computes `logprobs = logprobs_from_logits(logits[:, :-1], input_ids[:, 1:])`
- Builds `masks` that are 1 only for response tokens (0 for query tokens and padding)
- During this forward, hooks fire and populate `self._xs[name]`, `self._hs[name]` for each LoRA layer

**b) Ref model forward pass (no hooks, no grad)**
- Runs the reference model (frozen base without LoRA adapter) to get `tb_ref_logprobs`
- Used to compute the KL penalty

**c) Compute rewards and advantages**
- `compute_rewards`: adds the RM score at the last response token and subtracts `kl_ctl.value × KL(π || π_ref)` at every token → per-token reward tensor
- `compute_advantages`: runs GAE (Generalized Advantage Estimation) over the per-token rewards using `values` from the value head

**d) Ghost backward on full PPO loss**
```python
self._record_ghost = True
with ghost_mode(self.optimizer):   # optimizer.step = no-op
    self.train_minibatch(logprobs_frozen, values_frozen, logprobs, logits, vpreds, masks, advantages, returns, retain_graph=True)
self._record_ghost = False
```
- `train_minibatch` calls `self.loss(...)` which computes:
  - Policy loss: `max(−A·ratio, −A·clip(ratio, 1−ε, 1+ε))` averaged over masked tokens
  - Value loss: `0.5 × max((vpred−ret)², (vpred_clipped−ret)²)` averaged over masked tokens
  - Combined: `loss = pg_loss + vf_coef × vf_loss`
- `accelerator.backward(loss, retain_graph=True)` runs backprop
- Hooks fire and populate `self._gAs[name]`, `self._gBs[name]` for each layer
- `ghost_mode` ensures `optimizer.step()` is skipped — no weight update

After the loop, training factors are saved:
```python
self._train_xs  = {k: torch.cat(v) for k, v in self._xs.items()}
self._train_hs  = {k: torch.cat(v) for k, v in self._hs.items()}
self._train_gAs = {k: torch.cat(v) for k, v in self._gAs.items()}
self._train_gBs = {k: torch.cat(v) for k, v in self._gBs.items()}
```
These tensors have shape `[N=256, S, d]` per layer.

---

### Phase 3: Validation Ghost Backward — Compute influence scores per chunk

For each validation chunk of size `tracin_val_batch_size=64` (16 iterations for 1024 val samples):

**a) Clear hook buffers** (so current val chunk starts clean)

**b) Slice inputs from the globally-padded `val_model_inputs`**
```python
tracin_batch_inds = np.arange(vb_start, vb_end)
val_tracin_model_inputs = {k: val_model_inputs[k][tracin_batch_inds] for k in model_inputs_names}
```
All chunks have the same sequence length `S` because they came from one global collation.

**c) Forward pass on val chunk (with hooks ON)**
- Populates `self._xs[name]`, `self._hs[name]` for this val chunk

**d) Ref forward + rewards + advantages** (same pattern as training phase)

**e) Compute validation loss** (controlled by `--val_loss_type`):
- `seqloss-lastadv`: `val_loss = mean(−seq_logprob × last_advantage)` where `seq_logprob = sum_t logprob_t` over response tokens
- This gives a reward-weighted signal telling: "how much does the policy improve on this val sample?"

**f) Ghost backward on val loss**
```python
self._record_ghost = True
self.accelerator.backward(validation_loss)
self._record_ghost = False
self.optimizer.zero_grad()
```
- Populates `self._gAs[name]`, `self._gBs[name]` with the **validation gradient factors** for this chunk

**g) Compute influence scores for all 256 training samples vs. this val chunk**
```python
ghost_ip = self.compute_ghost_inner_product_diff_train_val_matrix_op()
sum_ghost_ip += ghost_ip
```

---

### `compute_ghost_inner_product_diff_train_val_matrix_op` — The influence score computation

For each LoRA layer `l`:

**Training factors** (256 samples, stored in `self._train_gAs/gBs/xs/hs`):
- `Q_A[i] = gA_train[i].T @ x_train[i]`  → shape `[d, D]`  (the full gradient matrix ∇A for sample i, implicit)

**Validation factors** (current chunk of 64, stored in `self._gAs/gBs/xs/hs`):
- `P_A[j] = gA_val[j].T @ x_val[j]`  → shape `[d, D]`
- `S_A = sum_j P_A[j]`  → the aggregated val gradient across the 64 val samples, shape `[d, D]`

**Inner product for each training sample `i`:**
```
IP_A[i] = (Q_A[i] * S_A).sum()  →  scalar
IP_B[i] = (Q_B[i] * S_B).sum()  →  scalar
IP[i]   = IP_A[i] + IP_B[i]
```

This computes `∇_θ L_train[i] · ∇_θ L_val[chunk]` using the factored LoRA representation — **without ever materializing the full gradient vectors**.

After 16 val chunks:
```
sum_ghost_ip[i] = Σ_chunk IP[i, chunk] = ∇_θ L_train[i] · ∇_θ L_val[all 1024]
```

---

### Phase 4: Sample Selection

```python
selected_ids = np.where(np.array(ghost_ip) > 0)[0]
```
Keeps only training samples with **positive influence** — their gradient points in the same direction as the validation gradient (i.e., their inclusion helps validation performance).

---

### Phase 5: PPO Optimization on Selected Samples

`model.train()` is called.

Runs `ppo_epochs=4` epochs over the `selected_ids`:
- Random permutation of selected indices each epoch
- For each mini-batch of size `mini_batch_size=1`:
  - Fresh forward pass: `batched_forward_pass(model, queries[i], responses[i], inputs[i])`
  - `train_minibatch(old_logprobs, old_values, new_logprobs, logits, vpreds, masks, advantages, returns)`
  - This time `ghost_mode` is **NOT** used → `optimizer.step()` runs and weights are updated
  - Gradient clipping: `clip_grad_norm_(model_params, max_grad_norm)`

**`loss` function** (called inside `train_minibatch`):
- Importance ratio: `ratio = exp(logprobs_new − logprobs_old)`
- Safety check: if `mean(ratio) > ratio_threshold=10.0`, skip this mini-batch (zero loss)
- `pg_loss = mean(max(−A·ratio, −A·clip(ratio, 1−ε, 1+ε)))` over masked tokens
- `vf_loss = 0.5 × mean(max((vpred−ret)², (vpred_clip−ret)²))` over masked tokens
- `total = pg_loss + 0.1 × vf_loss`

### Phase 6: KL Update, Stats, Cleanup

- `self.kl_ctl.update(kl, batch_size)` — adapts the KL coefficient for next step
- `record_step_stats` — computes mean rewards, KL, entropy, clip fractions, etc.
- Hook buffers are cleared: `self._xs/hs/gAs/gBs = {name: [] for ...}`
- `torch.cuda.empty_cache()`

---

## 6. Key Data Flow Summary

```
run_train_iif.sh
  └─ train_rlhf.py
       ├─ load_models()         → PPOTrainer (registers hooks)
       └─ train_loop_with_validation()
            ├─ get_rollouts()   → 256 train + 1024 val responses
            ├─ process_reward() → RM scores
            └─ step_with_validation()
                 ├─ [PHASE 1] global collation (one prepare_model_inputs per set)
                 ├─ [PHASE 2] 4× ghost backward (PPO loss) → _train_xs/hs/gAs/gBs
                 ├─ [PHASE 3] 16× val chunk loop:
                 │    ├─ ghost backward (val loss) → _xs/hs/gAs/gBs
                 │    └─ compute_ghost_inner_product_diff_train_val_matrix_op()
                 │         → per-sample IP scores (accumulated)
                 ├─ [PHASE 4] select samples: ghost_ip > 0
                 └─ [PHASE 5] 4 PPO epochs, real optimizer.step()
```

---

## 7. Important Design Choices

| Choice | Reason |
|---|---|
| `ghost_mode` (no-op optimizer.step) | Run backward to capture gradient factors without changing weights |
| LoRA hook factors (4 tensors per layer) | Avoid storing full gradient vectors; reconstruct dot-products on-the-fly |
| Global val padding (one collation upfront) | Ensure every val chunk has the same sequence length, so gradient contributions are comparable across chunks |
| `retain_graph=True` in ghost backward | Keep the computation graph so the val backward can reuse activations (in the single-pass `step_part_I` variant) |
| `model.eval()` + dropout disabled | Deterministic gradients for reproducible influence scores |
| `seqloss-lastadv` validation loss | Reward-weighted signal that captures "does this val sample benefit from better policy?" |
