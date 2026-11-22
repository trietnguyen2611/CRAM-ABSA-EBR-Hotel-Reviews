# CRAM-ABSA Notebooks, Papers & Pipeline Index

> Structural index of the CRAM-ABSA repository (`/Users/trietnguyen/Documents/ABSA`) so `graphify` links Jupyter notebooks (`.ipynb`), LaTeX papers (`.tex`), Python modules, and Electron desktop code into one navigable knowledge graph.

## 1. End-to-End Pipeline Architecture

```text
data/dataset/TripAdvisor_EN.json
  └─▶ Project.ipynb (Data Cleaning, EDA, 20 Features, SQLite + Parquet export)
        ├─▶ data/df/hotel_reviews.parquet (9,990 reviews × 42 cols: train/val/test)
        ├─▶ data/df/chatbot.db
        └─▶ docs/main_paper/chart_img/* & docs/main_pp_sci27/chart_img/*
              │
              ├─▶ train/train_model/trainv1.ipynb ──▶ model/modelv1/modelv1.pt
              ├─▶ train/train_model/trainv2.ipynb ──▶ model/modelv2/modelv2.pt (+ last_checkpoint_v2.pt, ebr_exemplar_bank_v2.pkl, training_history_v2.json)
              └─▶ train/train_method/trainmethod.ipynb ──▶ model/method/modelmethod{1,4,9}.pt
                    │
                    ├─▶ benchmark/benchmark.py (Web GUI on port 8765 + v1/v2/baseline evaluation)
                    └─▶ app/desktop/ (app.py HTTP backend + main.cjs Electron + app.js/index.html/style.css) & app/new/
```

---

## 2. Jupyter Notebooks (`.ipynb`)

### 2.1 `Project.ipynb` — Dataset Ingestion, EDA & 20-Feature Engineering (96 cells)
- **Sections**:
  - `0–1`: Environment initialization & loading `data/dataset/TripAdvisor_EN.json`.
  - `2–3`: DataFrame creation, SQL queries for Research Questions (RQs), integrity checks, text length quantiles.
  - `4`: Visualization & paper figures (`draw_rounded_card`, `build_figure_411`, `build_figure_412`, `generate_figure_413`).
  - `5`: Dual-stream NLTK preprocessing (`nltk_negation_aware_preprocess`, `preprocess_text_paper1`, `infer_aspect_sentiment_from_text_nltk`, `compute_aspect_polarity`), 20 tabular feature extractors (`calc_feat_aspect_count`, `calc_feat_dominant_aspect`, `calc_feat_net_sentiment_score`, `calc_feat_sentiment_conflict`, `calc_feat_evidence_density`, `calc_feat_avg_rationale_len`, `calc_feat_has_cross_overlap`, `calc_feat_neutral_hedging_density`, `calc_feat_lexical_polarity_intensity`, `calc_feat_nltk_clean_token_ratio`, `calc_feat_nltk_opinion_negation_density`, `calc_feat_ebr_aspect_prototype_sim`), Parquet + SQLite export (`data/df/hotel_reviews.parquet`, `data/df/chatbot.db`), multi-review synthesis (`demonstrate_multi_review_aspect_synthesis`), and span cleaning (`clean_annotation_spans`, `clean_hotel_name`).

### 2.2 `train/train_model/trainv1.ipynb` — CRAM-ABSA v1 Training (18 cells)
- **Sections & Symbols**:
  - `1`: `TrainConfig` (produces `model/modelv1/modelv1.pt`, `ebr_exemplar_bank.pkl`, `training_history_v1.json`).
  - `2`: Dual-stream NLTK & 20 features (`nltk_negation_aware_preprocess`, `preprocess_text_paper1`, `parse_annotations_safe`, `split_into_opinion_clauses`, `compute_span_calibrated_sentiment`, `extract_20_features_for_new_review`).
  - `3–4`: `EBRExemplarBank`, `derive_review_sentiment_id`, `cached_lemmatize`, `CRAMABSADataset`.
  - `5–6`: `CRAM_ABSA_MultiAspectModel`, `CRAMABSALoss`, `decode_corn_predictions`.
  - `7–8`: `evaluate_cram_absa`, `generate_structured_review_output`, `load_trained_model_and_predict`.

### 2.3 `train/train_model/trainv2.ipynb` — CRAM-ABSA v2 Training (MacBook Air M3 8GB MPS Optimized, 16 cells)
- **Sections & Symbols**:
  - `0–1`: MPS memory watermarks (`PYTORCH_MPS_HIGH_WATERMARK_RATIO="0.80"`, `LOW="0.55"`), `torch.mps.set_per_process_memory_fraction(0.80)` + startup `MPS recommended_max`/`driver_allocated` telemetry, `resolve_repo_root`, `TrainConfig` (`BACKBONE_NAME="BAAI/bge-small-en-v1.5"`, `HIDDEN_DIM=384`, `PROJ_DIM=256`, `ATTN_HEADS=6`, `MAX_SEQ_LEN=192`, `MAX_COMMENT_LEN=48`, `COMMENTARY_CHUNK_SIZE=128`, `PER_GPU_BATCH_SIZE=8` on MPS / `16` else, `GRAD_ACCUM_STEPS=4` on MPS / `2` else, effective batch `32`, `EVAL_BATCH_SIZE=16` on MPS, `AMP_DTYPE=torch.bfloat16` on MPS, `USE_GRAD_CHECKPOINTING=(DEVICE.type == "mps")`, `USE_DYNAMIC_PADDING=True`, `PAD_TO_MULTIPLE_OF=32`, `USE_LENGTH_BUCKETING=True`, `LENGTH_BUCKET_MULT=32`, `MPS_EMPTY_CACHE_EVERY=50`, `MPS_DRIVER_FLUSH_GB=2.2`, `EPOCHS=10`, `LR_BACKBONE=4e-5`, `LR_HEADS=1.5e-4`, `LLRD_DECAY=0.92`, `WEIGHT_DECAY=0.01`, `WARMUP_RATIO=0.10`, `LAST_CKPT_PATH=last_checkpoint_v2.pt`, `EARLY_STOPPING_PATIENCE=3`).
  - `2`: Hybrid clause splitter & 20-feature extraction (`_count_negated_lexicon`, `compute_span_calibrated_sentiment`, `_trim_span_offsets`, `split_into_opinion_clauses`, `count_aspect_keyword_hits`, `extract_20_features_for_new_review`).
  - `3`: Parquet loading, `EBRExemplarBank`, and explicit `del df_master; gc.collect()`.
  - `4`: Stale dataset/loader global pop + `gc.collect()`, compact CPU-dtype `CRAMABSADataset` (`int32`/`uint8`/`int8`/`float32`, `padding=False`, skipping eager `test_dataset` construction), `LengthBucketBatchSampler` (deterministic per-epoch mega-bucket shuffling via `set_epoch` on train; descending length-sorted batches on val), and `make_cram_collate_fn` (dynamic multiple-of-32 padding via `PAD_TO_MULTIPLE_OF=32`).
  - `5–6`: `CRAM_ABSA_MultiAspectModel` (dynamic `backbone_name`, `self.hidden_dim`/`self.attn_heads` derived from `self.bert.config`, `attn_implementation="sdpa"` on MPS, `self.bert.get_input_embeddings()`, float32 post-BERT heads, 2-layer MLP `aspect_multi_pol_head` + `token_polarity_proj`, active-row & trimmed `c_eff` fp32 `commentary_gru` with checkpointed chunked CE at `COMMENTARY_CHUNK_SIZE=128`), `CRAMABSALoss`, `decode_corn_predictions`.
  - `7`: Synchronized `release_accelerator_memory()` (`torch.mps.synchronize()` + `empty_cache()`), `maybe_flush_mps_cache(step)` (`step % 50` or `get_mps_driver_gb() >= 2.2`), pre-init stale global cleanup before model creation, `EpochProgressBar` (`_mem_tag` reporting `MPS {alloc:.2f}/{drv:.2f}GB` + 1-line `[Train Start]` banner), `build_llrd_optimizer_groups` (`LLRD_DECAY=0.92`, zero weight decay on bias & LayerNorm/norm), freeing `df_train`/`df_val`/`ebr_bank` after `initialize_ebr_prototypes`, `@torch.inference_mode()` `evaluate_cram_absa` (skipping `commentary_target_ids` + `maybe_flush_mps_cache`), `atomic_torch_save` / `atomic_json_save` (persisting `backbone_name`, `hidden_dim`, `proj_dim`, `attn_heads`, `max_seq_len`), mismatch-archiving Auto-Resume (`last_checkpoint_v2_prev.pt.bak`), Early Stopping (`patience=3`), 3-strike MPS OOM recovery guards, and `KeyboardInterrupt` cleanup preserving `cfg.LAST_CKPT_PATH`; outputs `model/modelv2/modelv2.pt`.
- **Design Doc**: `docs/CRAM_ABSA_v2_Design.md`.

### 2.4 `train/train_method/trainmethod.ipynb` — Baseline & Ablation Methods (12 cells)
- **Sections & Symbols**:
  - `1`: `UnifiedBenchmarkConfig`, `parse_annotations_safe`, `derive_review_sentiment_id`.
  - `2`: Paper 1 (`modelmethod1.pt`) — `Paper1ABSADataset`, `Paper1BERTModel`, `paper1_clean_text`, `apply_random_oversampling`, `evaluate_bert_baseline`.
  - `3`: Paper 4 (`modelmethod4.pt`) — `paper4_preprocess_tokens`, `compute_bm25_score`, `assemble_aspect_sentences_bm25`, `build_sparse_tfigm_matrix`, `Paper4MultiRegionCNN`, `encode_seq_ids`, `evaluate_paper4`.
  - `4–5`: Paper 9 (`modelmethod9.pt`) — `enrich_text_with_wordnet_wsd`, `build_normalized_dependency_adj`, `Paper9WSDDataset`, `GraphConvLayer`, `Paper9WSD_BiLSTM_GCN`.

---

## 3. Evaluation Benchmark & Desktop Application

- **`benchmark/benchmark.py`**:
  - Dynamic v1 (`bert-base-uncased`, 768d) & v2 (`BAAI/bge-small-en-v1.5`, 384d, MLP `aspect_multi_pol_head`, `max_seq_len=192`) model loading and caching via `resolve_model_file`, `get_shared_bert_tokenizer` (`_CACHED_TOKENIZERS`), `load_model_bundle` (`MAX_CACHED_MODELS=1` on MPS else `2`, inferring `backbone_name`, `hidden_dim`, `proj_dim`, `attn_heads`, `use_mlp_multi_pol` via `.0.weight` vs `.weight`, `max_seq_len`, and stripping `model_state_dict` from cached `ckpt` metadata), `prepare_cram_test_tensors` (`_CACHED_TEST_CRAM_TENSORS` keyed by `(backbone_name, max_len)`), and `prepare_paper1_or_9_test_tensors`.
  - `CRAM_ABSA_MultiAspectModel` (`sdpa` on MPS, dynamic `hidden_dim`/`attn_heads`, `get_input_embeddings()`, `use_mlp_multi_pol`) and MPS fp32 autocast guards across CRAM-ABSA, `Paper1BERTModel`, `GraphConvLayer`, and `Paper9WSD_BiLSTM_GCN`.
  - Evaluates test split (`load_test_dataframe`, `run_test_split_benchmark`, `_benchmark_worker`, `start_benchmark_async`, `get_benchmark_data` with `torch.inference_mode()`, `_get_eval_amp_context()` `bfloat16` autocast on MPS, MPS `batch_size=16` for `<=384d` / `8` for `>384d`, dynamic per-batch trimming, periodic `torch.mps.empty_cache()`, and optional `multi_pol_f1` & `tok_pol_acc` metrics) and single-review inference (`evaluate_new_review`, `extract_multi_polarity_segments`, `format_aspect_segments_output`, `extract_20_features_for_new_review`).
  - Serves Web GUI via `ThreadedHTTPServer` and `BenchmarkRequestHandler`.
- **`app/desktop/`** (and `app/new/`):
  - Python backend (`app/desktop/app.py`): `load_hotel_database`, `RobustHotelMatcher`, `MultiReviewRetriever`, `BertMultiTaskQA`, `HotelAspectQA_GenerativeBERT`, `GenerativeQAPipeline`, `SessionManager`, `ConversationalQABot`, `ThreadedHTTPServer`, `QAChatHandler`.
  - Electron & UI shell: `app/desktop/main.cjs`, `app/desktop/app.js`, `app/desktop/index.html`, `app/desktop/style.css`, `app/desktop/DESIGN-apple.md`, `app/desktop/package.json`.

---

## 4. Academic Papers & Project Customizations

- **LaTeX Manuscripts**:
  - `docs/main_paper/sn-article.tex` (`sn-jnl`) & `docs/main_pp_sci27/sn-article.tex` (`llncs` for SCI-2027): *CRAM-ABSA: Multi-Task Rationale Alignment for Aspect-Based Hotel Review Assessment* (Introduction & RQ1–RQ3, Related Work, Proposed Methodology Stages 1–2, Experimental Evaluation, Discussion & Conclusion), compiled via `build.sh` with `sn-bibliography.bib` and 30 PDF references in `docs/src_paper/`.
- **Agent & Graphify Configuration**:
  - `.agents/rules/graphify.md`: Mandatory Graphify-first query & post-edit update protocol for coordinator and all swarm agents.
  - `.agents/skills/graphify/SKILL.md`: `/graphify` workflow skill for rebuilding, querying, and syncing `graphify-out/`.
  - `.agents/skills/absa-swarm/SKILL.md`: `/swarm` 6-specialist agent topology (`data-engineer`, `ml-trainer`, `benchmark-engineer`, `app-engineer`, `paper-docs`, `qa-verifier`, plus `monitor`).
