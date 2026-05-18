# DataInf PPO Workflow

This document traces the DataInf pipeline from launch script to final weight update, file by file and function by function.

---

## Background: How DataInf Differs from TracIn

| Aspect | TracIn (IIF) | DataInf |
|---|---|---|
| Influence formula | `∇L_train[i] · ∇L_val` (dot product) | Sherman-Morrison inversion: `v^T H^{-1} g` |
| # ghost backward passes per training mini-batch | 1 (full PPO loss) | 2 (unweighted for Hessian + full PPO for influence gradient) |
| Hessian | Ignored (identity approximation) | Explicitly approximated via rank-1 outer products |
| Damping `λ` | None | Single global term absorbing curvature scale |
| Weight clipping | None | Percentile-based clipping of `w_i` to maintain PSD |

---

## 1. Entry Point — `scripts/run_train_datainf.sh`

The shell script launches training with these key flags:

```
--datainf               # activate DataInf mode (routes to step_datainf)
--val_loss_type=seqloss-lastadv
--batch_size=256        # N training samples per step
--tracin_batch_size=64  # ghost-gradient mini-batch size
--tracin_val_batch_size=64
--val_size=1024         # M validation samples
--ppo_epochs=4
--mini_batch_size=1
--datainf_damping_scale=1.0    # c_λ for adaptive damping
--datainf_percentile=2.0       # bottom 2% of w_i excluded from Hessian
--datainf_eps=1e-8             # positivity buffer for shift constant c
--ratio_threshold=10.0         # skip mini-batches with avg ratio > 10
```

---

## 2. `scripts/train_rlhf.py` — Setup and Routing

**What it does:**
- Parses all CLI args into `ScriptArguments` via `HfArgumentParser`
- Loads the training dataset and the **separate validation dataset**
  - For imdb: condition is `if script_args.with_validation or script_args.datainf:` — DataInf always needs validation data
- Creates `PPOTrainer(config, model, ref_model=None, tokenizer, dataset, data_collator, optimizer)`
  - `PPOTrainer.__init__` registers **forward and backward hooks** on every LoRA layer (see Section 4)
- Routes to `train_loop_with_validation(..., use_datainf=True)` when `--datainf` is set

---

## 3. `rlhfutils/rl_utils.py` → `train_loop_with_validation` — Outer Training Loop

Runs for `steps` iterations. This function is **shared** between TracIn and DataInf; the `use_datainf` flag decides which step function is called.

### 3a. Response Generation — `get_rollouts`
- Calls `ppo_trainer.generate(question_tensors, ...)` for **training questions** → `response_tensors`
- Calls `ppo_trainer.generate(val_question_tensors, ...)` for **validation questions** → `val_response_tensors`
- Both use the current policy with `torch.no_grad()`

### 3b. Reward Scoring — `process_reward`
- Runs the reward model on formatted `(query, response)` texts for both training and validation sets
- Returns scalar rewards per sample, converted to `torch.tensor` on device

### 3c. Dispatch to `ppo_trainer.step_datainf`
```python
stats = ppo_trainer.step_datainf(
    question_tensors, response_tensors, rewards,
    val_question_tensors, val_response_tensors, val_rewards,
    timing, gen_data_dir=...
)
```

---

## 4. `scripts/ppo_trainer.py` — Hook Registration (done once in `__init__`)

Persistent hooks are registered on every LoRA layer and on `v_head.summary` **once at construction time**, before any step is called. They fire only when `self._record_ghost = True`.

**Forward hook on `lora_A`:**
- Captures input `x` (shape `[B, S, d_in]`) into `self._xs[name]`
- Captures output `h = lora_A(x)` (shape `[B, S, r]`) into `self._hs[name]`

**Backward hook on `lora_B`:**
- Captures `g_h` (gradient w.r.t. lora_B's input, shape `[B, S, r]`) into `self._gAs[name]`
- Captures `g_o` (gradient w.r.t. lora_B's output, shape `[B, S, d_out]`) into `self._gBs[name]`

**Why these four quantities?**
For a LoRA layer `W = BA`, the full gradient matrix is a Kronecker outer product:
```
∇_A L_i  =  g_h_i^T ⊗ x_i    (r × d_in)
∇_B L_i  =  g_o_i^T ⊗ h_i    (d_out × r)
```
Storing the four smaller factors (`x`, `h`, `g_h`, `g_o`) avoids materializing full gradient matrices while still allowing exact inner-product reconstruction.

**`ghost_mode` context manager (`scripts/ppo_trainer.py`, lines ~106–113):**
Temporarily replaces `optimizer.step` with a no-op so a backward pass captures gradients **without updating weights**.

---

## 5. `scripts/ppo_trainer.py` → `step_datainf` — Per-Step Logic

This is the core function for DataInf. It has 6 phases.

---

### Phase 1: Global Training Collation + TWO Ghost Backward Passes

**Collation (done once for full training batch):**
```python
model_inputs = self.prepare_model_inputs(queries, responses)
```
Pads the full 256-sample training batch to the **global max sequence length** using the tokenizer's `pad_token_id`. If distributed, `pad_across_processes` aligns across GPUs.

`model.eval()` and all dropout modules set to eval mode for deterministic gradients.

**Two separate accumulator dicts are initialized:**
```python
hessian_gAs_accum = {}   # from unweighted backward → h_{i,l}
hessian_gBs_accum = {}
ppo_gAs_accum = {}       # from full PPO backward → g_k^{PPO}
ppo_gBs_accum = {}
```

**Loop over 4 training mini-batches (64 samples each):**

For each mini-batch `[tb_start : tb_end]`:

**a) Forward pass (hooks ON)** — `batched_forward_pass`
```python
self._record_ghost = True
tb_logprobs, tb_logits, tb_values, tb_masks = self.batched_forward_pass(...)
self._record_ghost = False
```
Populates `self._xs[name]`, `self._hs[name]` for all LoRA layers.
`batched_forward_pass` runs `model(**inputs)` → returns logits, values; computes `logprobs = logprobs_from_logits(logits[:,:-1], input_ids[:,1:])`; builds response-only masks.

**b) Ref model forward pass (no hooks, no grad)**
Runs with LoRA adapter disabled (`disable_adapter`) to get `tb_ref_logprobs` for KL calculation.

**c) Compute rewards and advantages (no grad)**
- `compute_rewards`: adds RM score at last response token; subtracts `kl_ctl.value × KL` at every token
- `compute_advantages`: runs GAE (Generalized Advantage Estimation) using per-token rewards + value estimates → `advantages`, `returns`

---

**Ghost backward #1: UNWEIGHTED log-prob (for Hessian `h_{i,l}`)**

```python
# Clear buffers first to avoid contamination from previous iteration's PPO backward
for name in self._gAs:
    self._gAs[name] = []
    self._gBs[name] = []

unweighted_loss = -(tb_logprobs.to(torch.float32) * tb_masks.detach().float()).sum()

self._record_ghost = True
with ghost_mode(self.optimizer):           # optimizer.step = no-op
    self.accelerator.backward(unweighted_loss, retain_graph=True)
self.optimizer.zero_grad()
self._record_ghost = False

# Save these into hessian_gAs_accum / hessian_gBs_accum
```

**Why unweighted?** The Hessian is approximated as:
```
H_l ≈ (1/N*) Σ_i (w_i + c) · h_{i,l} h_{i,l}^T
```
To ensure this matrix is **Positive Semi-Definite (PSD)** (required for Sherman-Morrison inversion to be stable), the base gradient `h_{i,l}` must come from an unweighted, always-positive loss. Using the full PPO loss (which can have negative weighted contributions) would risk indefinite rank-1 terms.

---

**Ghost backward #2: FULL PPO loss (for influence gradient `g_k^{PPO}`)**

```python
# Clear buffers again so PPO backward starts fresh
for name in self._gAs:
    self._gAs[name] = []
    self._gBs[name] = []

self._record_ghost = True
with ghost_mode(self.optimizer):
    self.train_minibatch(
        old_logprobs=tb_logprobs.detach(),
        old_values=tb_values_upd.detach(),
        logprobs=tb_logprobs,
        logits=tb_logits,
        vpreds=tb_values,
        mask=tb_masks.detach(),
        advantages=tb_advantages,
        returns=tb_returns,
    )
self.optimizer.zero_grad()
self._record_ghost = False

# Save these into ppo_gAs_accum / ppo_gBs_accum
```

`train_minibatch` calls `self.loss(...)` which computes the full PPO loss:
- Policy loss: `mean(max(−A·ratio, −A·clip(ratio, 1−ε, 1+ε)))` over masked tokens
- Value loss: `0.5 × mean(max((vpred−ret)², (vpred_clipped−ret)²))` over masked tokens
- Combined: `loss = pg_loss + vf_coef × vf_loss`

**Why full PPO loss here?** The influence formula's loss gradient term `g_k^{PPO}` should capture what the full PPO training signal looks like for each sample. Using a simplified gradient would produce influence scores that don't reflect the actual training dynamics.

**After all mini-batches:** concatenate accumulators:
```python
train_xs = {k: torch.cat(v) ...}     # [N, S, d_in] per layer
train_hs = {k: torch.cat(v) ...}     # [N, S, r] per layer
hessian_gAs/gBs                       # from unweighted backward
ppo_gAs/gBs                           # from full PPO backward
```

---

### Phase 2: Effective Weights `w_i` + Percentile Clipping

```python
advantages = batch_dict["advantages"]   # [N, S]
masks      = batch_dict["masks"]        # [N, S]

# Sample-level advantage: average over response tokens
per_sample_adv = (advantages * masks).sum(dim=1) / masks.sum(dim=1).clamp(min=1)

# Effective weight: advantage minus KL coefficient, normalized by batch size
beta = self.kl_ctl.value
w = (per_sample_adv - beta) / N          # shape [N]
```

**Why this formula?** From the PPO linearized loss derivation, each sample's contribution to the objective is proportional to `(A_i - β) / N` where `A_i` is the sample's advantage and `β` is the current KL penalty coefficient.

**Percentile clipping:**
```python
p = self.config.datainf_percentile   # 2.0
Q_p = torch.quantile(w, p / 100.0)   # 2nd percentile of w

# Shift constant c ensures (w_i + c) > 0 for all retained samples
c_shift = max(0.0, -Q_p.item()) + self.config.datainf_eps

# Retain only samples where w_i >= Q_p (discard bottom 2%)
retained_mask = w >= Q_p
retained_ids  = torch.where(retained_mask)[0]   # shape [N*]
N_star = len(retained_ids)
w_retained = w[retained_ids]                     # shape [N*]
```

**Why clip?** Samples with very negative `w_i` would produce large negative diagonal Hessian entries `(w_i + c) · ‖h_i‖²`, destabilizing the Sherman-Morrison inversion. Excluding the bottom `p%` keeps the Hessian well-conditioned. The shift constant `c` ensures all retained `w_i + c > 0`, guaranteeing PSD.

**Subset all training factors to `retained_ids`:**
```python
ret_xs, ret_hs, ret_hessian_gAs/gBs, ret_ppo_gAs/gBs
```

---

### Phase 3: Validation Ghost Backward — Accumulate `v_l`

**Critical design: global-max padding**

Before the validation chunk loop, the **entire validation set is collated once**:
```python
val_model_inputs = self.prepare_model_inputs(val_queries, val_responses)
# + pad_across_processes if distributed
val_model_inputs_keys = list(val_model_inputs.keys())
```

This pads all 1024 validation samples to the **same sequence length** globally. Then inside the chunk loop, sub-batches are **sliced** from this single padded tensor:
```python
vb_inputs = {k: val_model_inputs[k][vb_start:vb_end] for k in val_model_inputs_keys}
```

**Why is global padding critical?** If instead you called `prepare_model_inputs` per chunk of 64, each chunk would be padded to its own local maximum length. This produces different sequence lengths across chunks, so the gradient contribution of each token position is scaled differently across chunks. When you accumulate `val_S_A += chunk_A` across 16 chunks, the accumulated gradient is biased by the varying sequence lengths. With global padding, every chunk is padded to the same length `S`, so gradients are comparable across all 16 chunks and the accumulation is unbiased.

**Loop over 16 validation chunks (64 samples each):**

For each chunk `[vb_start : vb_end]`:

**a) Clear hook buffers** for this chunk.

**b) Forward pass (hooks ON)** → `batched_forward_pass`
Captures `vb_xs[name]`, `vb_hs[name]` for this val chunk.

**c) Ref forward + rewards + advantages** (same pattern as training — no grad)
- `compute_rewards` → `vb_rewards`
- `compute_advantages` → `vb_advantages`

**d) Compute validation loss** using `val_loss_type='seqloss-lastadv'` (matches IIF):
```python
seq_logprob = (vb_logprobs.to(float32) * vb_masks).sum(dim=1)    # sum over tokens [B]
indices = argmax(vb_masks, dim=1) + sum(vb_masks, dim=1) - 1      # last response token index [B]
indices = indices.clamp(min=0, max=vb_advantages.size(1) - 1)     # safety clamp
seq_score = vb_advantages[arange(B), indices]                      # last token advantage [B]
val_loss = mean(-seq_logprob * seq_score)                          # scalar
```

This loss says: "how much does increasing the sequence log-prob help the final advantage?" — a reward-weighted signal identical to what IIF uses.

**e) Ghost backward on `val_loss`** (hooks ON, ghost_mode NOT used — we do want grads accumulated but zero them after)
```python
self._record_ghost = True
self.accelerator.backward(val_loss)
self._record_ghost = False
self.optimizer.zero_grad()
```
Populates `self._gAs[name]`, `self._gBs[name]` with validation gradient factors.

**f) Compute and accumulate chunk's contribution to `val_S_A`, `val_S_B`:**
```python
for name in self._xs:
    v_xs  = torch.cat(self._xs[name]).float()    # [chunk, S, d_in]
    v_hs  = torch.cat(self._hs[name]).float()    # [chunk, S, r]
    v_gAs = torch.cat(self._gAs[name]).float()   # [chunk, S, r]
    v_gBs = torch.cat(self._gBs[name]).float()   # [chunk, S, d_out]

    # Factored gradient: P_A[j] = gA[j].T @ x[j]  → [r, d_in]
    chunk_A = matmul(v_gAs.transpose(1,2), v_xs).sum(dim=0)   # sum over chunk samples → [r, d_in]
    chunk_B = matmul(v_gBs.transpose(1,2), v_hs).sum(dim=0)   # sum over chunk samples → [d_out, r]

    val_S_A[name] += chunk_A
    val_S_B[name] += chunk_B
```

After all 16 chunks:
```python
val_S_A[name] /= M    # M = 1024 — averaged over all val samples
val_S_B[name] /= M
```

`val_S_A[name]` is the **averaged validation gradient** for LoRA-A of layer `name`: shape `[r, d_in]`.

---

### Phase 4: Compute DataInf Influence Scores — `compute_datainf_influence`

**Function signature:**
```python
def compute_datainf_influence(
    self, train_xs, train_hs,
    hessian_gAs, hessian_gBs,   # from unweighted backward
    ppo_gAs, ppo_gBs,           # from full PPO backward
    val_S_A, val_S_B,            # averaged validation gradient
    w_retained, c, N_star
) → List[float]   # shape [N*]
```

**Step 1: Compute global `λ` (two-pass over layers)**

First pass: collect `L_ii` from all layers to compute a single global damping factor.

For each LoRA layer `l`:
```python
# Factored base gradients (from unweighted backward)
base_P_A = matmul(h_gAs.transpose(1,2), xs)   # [N*, r, d_in]
base_P_B = matmul(h_gBs.transpose(1,2), hs)   # [N*, d_out, r]
base_A_flat = base_P_A.reshape(N*, -1)          # [N*, r*d_in]
base_B_flat = base_P_B.reshape(N*, -1)          # [N*, d_out*r]

# Diagonal Hessian element for sample i at layer l:
# L_{l,ii} = (w_i + c) * ||h_{i,l}||^2
base_norms = (base_A_flat**2).sum(dim=1) + (base_B_flat**2).sum(dim=1)  # [N*]
L_ii = (w_retained + c) * base_norms                                       # [N*]

L_ii_all_layers += L_ii.sum()
L_count += 1
```

After all layers:
```python
lambda_l = datainf_damping_scale * L_ii_all_layers / (N_star * L_count)
lambda_l = max(lambda_l, 1e-12)
```

**Why global `λ`?** `λ` is a single scalar that regularizes the full Hessian `H ≈ λI + Σ_i rank1_i`. Using a consistent `λ` across all layers ensures each layer's influence contribution is on the same scale.

**Step 2: Compute per-sample influence scores**

Second pass over all layers, accumulating influence:

For each LoRA layer `l`:
```python
# Factored PPO gradients (from full PPO backward)
ppo_P_A = matmul(p_gAs.transpose(1,2), xs)   # [N*, r, d_in]
ppo_P_B = matmul(p_gBs.transpose(1,2), hs)   # [N*, d_out, r]

# Validation inner products:
#   val_base_ip[i] = h_{i,l} · v_l   (base grad dot validation grad)
#   val_ppo_ip[k]  = g_k^{PPO} · v_l (PPO grad dot validation grad)
val_base_ip = (base_P_A * v_A).sum((1,2)) + (base_P_B * v_B).sum((1,2))   # [N*]
val_ppo_ip  = (ppo_P_A * v_A).sum((1,2)) + (ppo_P_B * v_B).sum((1,2))    # [N*]

# Sherman-Morrison correction coefficient:
# alpha[i] = (w_i + c) * (v_l · h_{i,l}) / (lambda_l + L_{l,ii})
alpha = (w_retained + c) * val_base_ip / (lambda_l + L_ii)    # [N*]

# Cross-gram matrix: base_i · ppo_k  (i, k range over N*)
# This is the key DataInf quantity — mixes Hessian eigenvectors with PPO gradient
cross_gram = (base_A_flat @ ppo_A_flat.T) + (base_B_flat @ ppo_B_flat.T)   # [N*, N*]

# Sherman-Morrison correction per sample k:
# correction[k] = (1/N*) Σ_i alpha[i] * (h_{i,l} · g_k^{PPO})
correction = alpha @ cross_gram   # [N*]

# Full DataInf influence for layer l, sample k:
# I_l(k) = -(1/λ) [ v^T g_k^{PPO}  -  (1/N*) correction[k] ]
# Currently simplified to:
influence_l = -(1.0) * val_ppo_ip
influence += influence_l
```

> **Note:** In the current code the Sherman-Morrison correction term is commented out (`influence_l = -(1.0) * val_ppo_ip`). The full formula is implemented but disabled for the current experimental phase. When enabled, the full formula is:
> `influence_l = (-1/λ) * (val_ppo_ip - correction / N_star)`

**Mathematical basis:** The DataInf influence approximation comes from the influence function:
```
I(k) = -v^T H^{-1} g_k
```
where `H` is the Hessian of the PPO training loss and `H^{-1}` is approximated via the Sherman-Morrison formula applied to the rank-1 outer product decomposition of each sample's contribution to `H`.

**Returns:** `List[float]` of length `N*` — one scalar influence score per retained training sample.

---

### Phase 5: Save & Filter

```python
# Map N* influence scores back to all N=256 indices
full_ip = np.full(bs, -np.inf)
for local_idx, global_idx in enumerate(retained_ids.cpu().numpy()):
    full_ip[global_idx] = ghost_ip[local_idx]

# Select samples (currently: negative influence = harmful to validation performance)
selected_ids = np.where(full_ip < 0)[0]
```

Save to disk for analysis:
```python
torch.save({
    "queries": ..., "responses": ..., "scores": ...,
    "ip_scores": ghost_ip, "w": w.cpu().numpy(),
    "retained_ids": retained_ids.cpu().numpy(),
    "kl_ctl_value": ...
}, f'{gen_data_dir}/datainf_scores_{save_cnt}.pt')
```

Log to wandb: `n_selected`, `n_total`, `selection_ratio`, `n_retained`, `c_shift`.

---

### Phase 6: PPO Optimization on Selected Samples

`model.train()` is NOT explicitly called here — the model remains in eval (dropouts off). The optimization loop:

`ppo_epochs=4` epochs over `selected_ids`:
- Random permutation of selected indices each epoch
- For each mini-batch (size `mini_batch_size=1`):
  - Fresh forward pass: `batched_forward_pass(model, queries[mb_inds], responses[mb_inds], ...)`
  - `train_minibatch(old_logprobs, old_values, new_logprobs, logits, vpreds, masks, advantages, returns)`
  - This time `ghost_mode` is **NOT** used → `optimizer.step()` runs and weights are updated
  - Grad clipping: `clip_grad_norm_(model_params, max_grad_norm)`

**`loss` function** (called inside `train_minibatch`):
- `ratio = exp(logprobs_new − logprobs_old)`
- Safety: if `mean(ratio) > ratio_threshold=10.0`, skip this mini-batch (zero loss, no update)
- `pg_loss = mean(max(−A·ratio, −A·clip(ratio, 1−ε, 1+ε)))` over masked tokens
- `vf_loss = 0.5 × mean(max((vpred−ret)², (vpred_clip−ret)²))` over masked tokens
- `total = pg_loss + vf_coef × vf_loss`

### Phase 7: Stats & Cleanup

- `record_step_stats` → computes mean rewards, KL, entropy, clip fractions
- Additional DataInf stats added: `ppo/datainf/n_selected`, `n_total`, `selection_ratio`, `n_retained`, `c_shift`
- `kl_ctl.update(kl, batch_size)` — adapts KL coefficient for next step
- All hook buffers cleared: `self._xs/hs/gAs/gBs = {name: [] for ...}`
- `torch.cuda.empty_cache()`

---

## 6. Key Data Flow Summary

```
run_train_datainf.sh
  └─ train_rlhf.py
       ├─ load_models()            → PPOTrainer (registers hooks)
       └─ train_loop_with_validation(use_datainf=True)
            ├─ get_rollouts()      → 256 train + 1024 val responses
            ├─ process_reward()    → RM scores
            └─ step_datainf()
                 ├─ [PHASE 1] global training collation
                 │   4× mini-batch loop:
                 │    ├─ forward (hooks ON)  → xs, hs per layer
                 │    ├─ ref forward         → ref_logprobs
                 │    ├─ compute_rewards + compute_advantages
                 │    ├─ ghost backward #1 (unweighted loss)
                 │    │    └─ → hessian_gAs/gBs per layer
                 │    └─ ghost backward #2 (full PPO loss)
                 │         └─ → ppo_gAs/gBs per layer
                 ├─ [PHASE 2] compute w_i, percentile clip c,
                 │             retained_ids, subset all factors
                 ├─ [PHASE 3] global val collation (one prepare_model_inputs)
                 │   16× val chunk loop:
                 │    ├─ forward (hooks ON) → vb_xs, vb_hs
                 │    ├─ ref forward + rewards + advantages
                 │    ├─ compute seqloss-lastadv val_loss
                 │    ├─ ghost backward on val_loss → vb_gAs/gBs
                 │    └─ accumulate chunk_A / chunk_B into val_S_A/val_S_B
                 │   normalize: val_S_A /= M,  val_S_B /= M
                 ├─ [PHASE 4] compute_datainf_influence()
                 │    ├─ pass 1: compute global λ from all layers' L_ii
                 │    └─ pass 2: per layer → base_P, ppo_P, alpha, cross_gram → influence
                 ├─ [PHASE 5] map scores to full batch, filter selected_ids, save to disk
                 └─ [PHASE 6] 4 PPO epochs, real optimizer.step() on selected_ids
```

---

## 7. Important Design Choices

| Choice | Reason |
|---|---|
| Two ghost backward passes | Need two distinct gradient semantics: unweighted for PSD Hessian, full PPO for accurate influence direction |
| Unweighted log-prob for Hessian | Ensures `(w_i+c) h_i h_i^T` is PSD for all retained samples, making Sherman-Morrison well-defined |
| Full PPO loss for `g_k^{PPO}` | Matches the actual training signal, so influence scores reflect real training dynamics |
| Percentile clipping (bottom 2%) | Discards extreme outliers that would make Hessian rank-1 terms negative and destabilize inversion |
| Shift constant `c = max(0, -Q_p) + ε` | Guarantees `w_i + c > 0` for all retained samples without modifying the sign of positive weights |
| Global `λ` across all layers | Ensures consistent regularization scale so each layer contributes equally to influence |
| Global val padding (one collation upfront) | Every chunk has the same sequence length `S`, so gradient contributions are comparable across all 16 chunks; per-chunk padding would bias the accumulated `val_S_A/val_S_B` |
| `seqloss-lastadv` validation loss | Reward-weighted signal matching IIF, so the validation gradient points in the direction of improved policy performance |
| `retain_graph=True` in unweighted backward | Keeps the computation graph alive for the subsequent PPO backward in the same loop iteration |
| `model.eval()` + dropout off | Deterministic, reproducible gradients for stable influence scores |
| `ratio_threshold` guard in `loss` | Skips mini-batches with pathological importance ratios that would produce large unstable gradient updates |
