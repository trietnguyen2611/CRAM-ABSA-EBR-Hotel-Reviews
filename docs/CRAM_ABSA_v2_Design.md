# CRAM-ABSA v2 — Architecture & Design Specification

> **Branch:** `cram-absa-v2`  
> **Baseline (v1.0):** `main` / tag `v1.0`  
> **Training Notebook:** [trainv2.ipynb](file:///Users/trietnguyen/Documents/ABSA/train/train_model/trainv2.ipynb)  
> **Benchmark Evaluator:** [benchmark.py](file:///Users/trietnguyen/Documents/ABSA/benchmark/benchmark.py)

---

## 1. Motivation: From v1 to v2

In **CRAM-ABSA v1**, each of the 6 hotel aspects (`Service`, `Facility`, `Experience`, `Amenity`, `Loyalty`, `Branding`) is predicted with a single 4-class sentiment label (`None`, `Negative`, `Neutral`, `Positive`) and a single rationale mask. When a guest simultaneously praises and complains about the **same aspect** (e.g., *"the room was spacious, however the bathroom was dirty"* for `Facility`), v1 collapses both signals into a single label without separating the praise segment from the complaint segment.

**CRAM-ABSA v2** extends v1 with:
1. **Multi-Polarity Aspect Detection (`aspect_multi_pol_head`)**: A 2-layer MLP multi-label head (`HIDDEN_DIM*2 -> 256 -> 3`) predicting `[Negative, Neutral, Positive]` presence per aspect to detect mixed praise + complaint within the same aspect.
2. **Token-Level Polarity Projection (`token_polarity_proj`)**: A 3-class linear token-level polarity head (`HIDDEN_DIM -> 3`) trained with clause-level calibrated polarity labels.
3. **Hybrid Clause Splitter & Contrastive Inheritance (`extract_aspect_segments` / `extract_multi_polarity_segments`)**: Splits reviews into opinion clauses, inherits aspect relevance across contrast conjunctions (`but`, `however`, `though`), and separates praise vs. complaint spans per aspect at inference time.
4. **MacBook Air M3 (8GB RAM) MPS Optimization**: Uses `BAAI/bge-small-en-v1.5` (384d, 12-layer, 6-head cross-attention) with `attn_implementation="sdpa"` on MPS (falling back gracefully), `bfloat16` AMP on MPS with float32 post-BERT heads/cross-attention, `USE_GRAD_CHECKPOINTING=(DEVICE.type == "mps")` (active on `384d`), active-row trimmed checkpointed chunked `commentary_gru` CE (`COMMENTARY_CHUNK_SIZE=128`), `LengthBucketBatchSampler` + dynamic multiple-of-32 padding (`PAD_TO_MULTIPLE_OF=32`), `build_llrd_optimizer_groups` (LLRD `0.92`), MPS watermarks (`0.80`/`0.55`) + `torch.mps.set_per_process_memory_fraction(0.80)`, synchronized `release_accelerator_memory()` (`torch.mps.synchronize()` + `empty_cache()`), `maybe_flush_mps_cache(step)` (`MPS_EMPTY_CACHE_EVERY=50` or `MPS_DRIVER_FLUSH_GB=2.2`), pre-init stale global cleanup, 3-strike OOM recovery guards, and `KeyboardInterrupt` cleanup preserving `last_checkpoint_v2.pt`.

---

## 2. Architecture Comparison (v1 vs v2)

| Component | CRAM-ABSA v1 (`v1.0`) | CRAM-ABSA v2 (`cram-absa-v2`) |
| :--- | :--- | :--- |
| **Backbone** | `bert-base-uncased` (768d, 12L, 8-head cross-attn, `MAX_SEQ_LEN=256`) + 20 tabular features via Residual Gate | `BAAI/bge-small-en-v1.5` (384d, 12L, 6-head cross-attn, `PROJ_DIM=256`, `MAX_SEQ_LEN=192`) + 20 tabular features via Residual Gate + `attn_implementation="sdpa"` on MPS |
| **Aspect Sentiment** | 6 × 4-class MLP heads (`768*2 -> 256 -> 4`, `None/Neg/Neu/Pos`) | 6 × 4-class MLP heads (`HIDDEN_DIM*2 -> 256 -> 4`, parameterized dynamically by `self.hidden_dim=384`) |
| **Multi-Polarity Head** | — | 2-layer MLP `aspect_multi_pol_head` (`HIDDEN_DIM*2 -> 256 -> 3`, `LayerNorm + GELU + Dropout(0.20)`) predicting `[Neg, Neu, Pos]` presence per aspect (`benchmark.py` auto-detects `.0.weight` MLP vs legacy `.weight` linear via `use_mlp_multi_pol`) |
| **Token Polarity Head** | — | Linear `token_polarity_proj` (`HIDDEN_DIM -> 3`) predicting token-level `[Neg, Neu, Pos]` |
| **Commentary Decoder** | `commentary_gru` + tied `commentary_out` | Tied via `self.bert.get_input_embeddings()`, filtered to `active_rows` and trimmed to `c_eff`, run in fp32 on MPS with checkpointed chunked CE (`COMMENTARY_CHUNK_SIZE=128`, `reduction="sum"`) |
| **Clause-Level Labels** | Span-level sentiment only | `split_into_opinion_clauses` + `compute_span_calibrated_sentiment` per sub-clause for `token_polarity_labels` |
| **Loss Function (`CRAMABSALoss`)** | Aspect CE + CORN + Overlap + Rationale + Commentary | Adds `BETA_MULTI_POL * (loss_multi_pol + 0.5 * loss_tok_pol)` with `pos_weight=4.0` on multi-pol BCE and `label_smoothing=0.02` on token polarity CE |
| **Training Schedule & Memory** | 8 epochs, batch 16 × accum 2, standard `AdamW` | 10 epochs (`EARLY_STOPPING_PATIENCE=3`), `PER_GPU_BATCH_SIZE=8` × `GRAD_ACCUM_STEPS=4` (eff 32), `EVAL_BATCH_SIZE=16`, `LR_BACKBONE=4e-5`, `LR_HEADS=1.5e-4`, `build_llrd_optimizer_groups` (`LLRD_DECAY=0.92`, zero WD on bias/norm), `LengthBucketBatchSampler` (`LENGTH_BUCKET_MULT=32`), `USE_DYNAMIC_PADDING=True` (`PAD_TO_MULTIPLE_OF=32`), `USE_GRAD_CHECKPOINTING=(DEVICE.type == "mps")`, MPS watermarks `0.80`/`0.55` + `set_per_process_memory_fraction(0.80)`, synchronized `release_accelerator_memory()` + `maybe_flush_mps_cache` (`MPS_EMPTY_CACHE_EVERY=50`, `MPS_DRIVER_FLUSH_GB=2.2`), pre-init stale global cleanup, 3-strike MPS OOM guard, mismatch-archiving Auto-Resume (`last_checkpoint_v2_prev.pt.bak`), and clean `KeyboardInterrupt` checkpoint preservation |
| **Evaluation Metrics & Inference** | `val_loss`, `acc_at_0`, `macro_mae`, `aspect_macro_f1`, `rationale_f1`, `leakage_rate`; 1 rationale string per aspect | Adds `multi_pol_f1` (micro-F1) and `tok_pol_acc`; multi-segment list `[{"sentiment": ..., "text": ...}]` per aspect; `benchmark.py` adds dynamic backbone/cache handling (`_CACHED_TOKENIZERS`, `(backbone_name, max_len)`-keyed `_CACHED_TEST_CRAM_TENSORS`, `MAX_CACHED_MODELS=1` on MPS), `bfloat16` eval autocast, MPS batch size `16` (`<=384d`) / `8` (`>384d`) & dynamic trimming |

---

## 3. Loss Formulation (v2)

$$\mathcal{L}_{\text{v2}} = \mathcal{L}_{\text{aspect}} + \beta_{\text{corn}} \mathcal{L}_{\text{corn}} + \beta_{\text{overlap}} \mathcal{L}_{\text{overlap}} + \beta_{\text{rat}} \mathcal{L}_{\text{rat}} + \beta_{\text{comm}} \mathcal{L}_{\text{comm}} + \beta_{\text{mp}} \left(\mathcal{L}_{\text{multi\_pol}} + 0.5\,\mathcal{L}_{\text{tok\_pol}}\right)$$

Default weights in `TrainConfig`:
- `BETA_CORN = 0.20`
- `BETA_OVERLAP = 0.35`
- `BETA_RATIONALE = 0.25`
- `BETA_COMMENTARY = 0.05`
- `BETA_MULTI_POL = 0.15`
- `aspect_multi_pol` BCE `pos_weight = 4.0` (compensates for ~85–90% negative entries across `(B, 6, 3)` slots) and `label_smoothing = 0.02` on `token_polarity_proj` CE.

---

## 4. Current Status & Next Steps on `cram-absa-v2`

- **Code & Benchmark Synchronization:** Both [trainv2.ipynb](file:///Users/trietnguyen/Documents/ABSA/train/train_model/trainv2.ipynb) and [benchmark.py](file:///Users/trietnguyen/Documents/ABSA/benchmark/benchmark.py) support the upgraded `BAAI/bge-small-en-v1.5` (384d, `max_seq_len=192`, MLP `aspect_multi_pol_head`) v2 architecture alongside v1 (`bert-base-uncased`, 768d) backward compatibility.
- **Checkpoint State:** `model/modelv2/` currently contains an earlier Epoch 1 legacy `bert-base-uncased` run (`modelv2.pt`, `last_checkpoint_v2.pt`, `ebr_exemplar_bank_v2.pkl`, `training_history_v2.json`). On the next training run, `trainv2.ipynb` detects the backbone/hidden-dim mismatch, archives `last_checkpoint_v2.pt` to `last_checkpoint_v2_prev.pt.bak`, and trains cleanly from Epoch 1 on `BAAI/bge-small-en-v1.5`.

1. Execute [trainv2.ipynb](file:///Users/trietnguyen/Documents/ABSA/train/train_model/trainv2.ipynb) (up to 10 epochs with `EARLY_STOPPING_PATIENCE=3`) to produce the converged `BAAI/bge-small-en-v1.5` `model/modelv2/modelv2.pt` and `model/modelv2/training_history_v2.json`.
2. Benchmark `model/modelv2/modelv2.pt` in [benchmark.py](file:///Users/trietnguyen/Documents/ABSA/benchmark/benchmark.py) and compare `v2` metrics (`aspect_macro_f1`, `rationale_f1`, `leakage_rate`, `multi_pol_f1`, `tok_pol_acc`) against `v1.0` (`model/modelv1/modelv1.pt`) and baselines (`model/method/modelmethod{1,4,9}.pt`).
3. Integrate `modelv2.pt` and multi-polarity segment output into `app/desktop/` (and/or `app/new/`).
