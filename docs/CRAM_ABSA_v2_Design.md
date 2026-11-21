# CRAM-ABSA v2 — Architecture & Design Specification

> **Branch:** `cram-absa-v2`  
> **Baseline (v1.0):** `main` / tag `v1.0`  
> **Training Notebook:** [trainv2.ipynb](file:///Users/trietnguyen/Documents/DAP-Hotel-2/train/train_model/trainv2.ipynb)

---

## 1. Motivation: From v1 to v2

In **CRAM-ABSA v1**, each of the 6 hotel aspects (`Service`, `Facility`, `Experience`, `Amenity`, `Loyalty`, `Branding`) is predicted with a single 4-class sentiment label (`None`, `Negative`, `Neutral`, `Positive`) and a single rationale mask. When a guest simultaneously praises and complains about the **same aspect** (e.g., *"the room was spacious, however the bathroom was dirty"* for `Facility`), v1 collapses both signals into a single label without separating the praise segment from the complaint segment.

**CRAM-ABSA v2** extends v1 with:
1. **Multi-Polarity Aspect Detection (`aspect_multi_pol_head`)**: A 3-logit multi-label head (`Negative`, `Neutral`, `Positive`) per aspect to detect mixed praise + complaint within the same aspect.
2. **Token-Level Polarity Projection (`token_polarity_proj`)**: A 3-class token-level polarity head trained with clause-level calibrated polarity labels.
3. **Hybrid Clause Splitter & Contrastive Inheritance (`extract_aspect_segments`)**: Splits reviews into opinion clauses, inherits aspect relevance across contrast conjunctions (`but`, `however`, `though`), and separates praise vs. complaint spans per aspect at inference time.
4. **Apple Silicon MPS + CoreML ANE Support**: `eager` attention fallback on MPS, float32 GRU + tied embedding projection under `autocast(enabled=False)` on MPS, and optional CoreML `.mlpackage` export.

---

## 2. Architecture Comparison (v1 vs v2)

| Component | CRAM-ABSA v1 (`v1.0`) | CRAM-ABSA v2 (`cram-absa-v2`) |
| :--- | :--- | :--- |
| **Backbone** | `bert-base-uncased` + 20 tabular features via Residual Gate | Same + `attn_implementation="eager"` on MPS |
| **Aspect Sentiment** | 6 × 4-class MLP heads (`None/Neg/Neu/Pos`) | Same |
| **Multi-Polarity Head** | — | 2-layer MLP (`HIDDEN_DIM*2 -> 256 -> 3`) predicting `[Neg, Neu, Pos]` presence per aspect |
| **Token Polarity Head** | — | Linear (`HIDDEN_DIM -> 3`) predicting token-level `[Neg, Neu, Pos]` |
| **Clause-Level Labels** | Span-level sentiment only | `split_into_opinion_clauses` + `compute_span_calibrated_sentiment` per sub-clause for `token_polarity_labels` |
| **Loss Function** | Aspect CE + CORN + Overlap + Rationale + Commentary | Adds `BETA_MULTI_POL * (loss_multi_pol + 0.5 * loss_tok_pol)` with `pos_weight=4.0` on multi-pol BCE and `label_smoothing=0.02` on token polarity CE |
| **Evaluation Metrics** | `val_loss`, `acc_at_0`, `macro_mae`, `aspect_macro_f1`, `rationale_f1`, `leakage_rate` | Adds `multi_pol_f1` (micro-F1) and `tok_pol_acc` |
| **Inference Output** | 1 sentiment + 1 rationale string per aspect | Multi-segment list `[{"sentiment": ..., "text": ...}]` per aspect |

---

## 3. Loss Formulation (v2)

$$\mathcal{L}_{\text{v2}} = \mathcal{L}_{\text{aspect}} + \beta_{\text{corn}} \mathcal{L}_{\text{corn}} + \beta_{\text{overlap}} \mathcal{L}_{\text{overlap}} + \beta_{\text{rat}} \mathcal{L}_{\text{rat}} + \beta_{\text{comm}} \mathcal{L}_{\text{comm}} + \beta_{\text{mp}} \left(\mathcal{L}_{\text{multi\_pol}} + 0.5\,\mathcal{L}_{\text{tok\_pol}}\right)$$

Default weights in `TrainConfig`:
- `BETA_CORN = 0.20`
- `BETA_OVERLAP = 0.35`
- `BETA_RATIONALE = 0.25`
- `BETA_COMMENTARY = 0.05`
- `BETA_MULTI_POL = 0.15`
- `aspect_multi_pol` BCE `pos_weight = 4.0` (compensates for ~85–90% negative entries across `(B, 6, 3)` slots).

---

## 4. Next Steps on `cram-absa-v2`

1. Run [trainv2.ipynb](file:///Users/trietnguyen/Documents/DAP-Hotel-2/train/train_model/trainv2.ipynb) to produce `model/modelv2/modelv2.pt` and `model/modelv2/training_history_v2.json`.
2. Compare `v2` metrics (`aspect_macro_f1`, `rationale_f1`, `leakage_rate`, `multi_pol_f1`, `tok_pol_acc`) against `v1.0`.
3. Once validated, integrate `modelv2.pt` into `benchmark/benchmark.py` and `app/`.
