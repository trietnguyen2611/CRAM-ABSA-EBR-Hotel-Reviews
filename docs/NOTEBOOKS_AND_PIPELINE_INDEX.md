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
                    └─▶ app/desktop/ (app.py HTTP backend + main.cjs Electron + app.js/index.html/style.css)
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

### 2.3 `train/train_model/trainv2.ipynb` — CRAM-ABSA v2 Training (MacBook Air M3 8GB MPS Optimized, 20 cells)
- **Sections & Symbols**:
  - `0–1`: MPS memory watermarks (`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7`, `LOW=0.5`), `resolve_repo_root`, `TrainConfig` (`MAX_SEQ_LEN=192`, `PER_GPU_BATCH_SIZE=4`, `GRAD_ACCUM_STEPS=8`, `EVAL_BATCH_SIZE=8`, `AMP_DTYPE=torch.bfloat16`, `USE_GRAD_CHECKPOINTING=True`, `USE_DYNAMIC_PADDING=True`, `LAST_CKPT_PATH=last_checkpoint_v2.pt`, `EARLY_STOPPING_PATIENCE=3`).
  - `2`: Hybrid clause splitter & 20-feature extraction (`_count_negated_lexicon`, `compute_span_calibrated_sentiment`, `_trim_span_offsets`, `split_into_opinion_clauses`, `count_aspect_keyword_hits`, `extract_20_features_for_new_review`).
  - `3`: Parquet loading, `EBRExemplarBank`, and explicit `del df_master; gc.collect()`.
  - `4`: Compact CPU-dtype `CRAMABSADataset` (`int32`/`uint8`/`int8`/`float32`, `padding=False`) + `make_cram_collate_fn` (dynamic multiple-of-8 padding).
  - `5–6`: `CRAM_ABSA_MultiAspectModel` (SDPA + gradient checkpointing on MPS, float32 post-BERT/GRU safety, multi-polarity & token-polarity heads), `CRAMABSALoss`, `decode_corn_predictions`.
  - `7`: `EpochProgressBar` (in-place 1%-milestone live progress bar without `ipywidgets`), `AdamW(..., foreach=False)`, `@torch.inference_mode()` `evaluate_cram_absa` (skipping `commentary_target_ids`), `atomic_torch_save`, `atomic_json_save`, Auto-Resume from `last_checkpoint_v2.pt`, Early Stopping (`patience=3`), outputs `model/modelv2/modelv2.pt`.
  - `8–9`: Multi-segment inference (`generate_structured_review_output`, `load_trained_model_and_predict` with `enable_grad_ckpt=False`) and CoreML ANE `.mlpackage` export (`CRAM_ABSA_ANE_Wrapper` with `attn_implementation="eager"`).
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
  - Loads checkpoints via `resolve_model_file`, `load_model_bundle`, `prepare_cram_test_tensors`, `prepare_paper1_or_9_test_tensors`.
  - Evaluates test split (`load_test_dataframe`, `run_test_split_benchmark`, `_benchmark_worker`, `start_benchmark_async`, `get_benchmark_data`) and single-review inference (`evaluate_new_review`, `extract_multi_polarity_segments`, `format_aspect_segments_output`, `extract_20_features_for_new_review`).
  - Serves Web GUI via `ThreadedHTTPServer` and `BenchmarkRequestHandler`.
- **`app/desktop/`**:
  - Python backend (`app/desktop/app.py`): `load_hotel_database`, `RobustHotelMatcher`, `MultiReviewRetriever`, `BertMultiTaskQA`, `HotelAspectQA_GenerativeBERT`, `GenerativeQAPipeline`, `SessionManager`, `ConversationalQABot`, `ThreadedHTTPServer`, `QAChatHandler`.
  - Electron & UI shell: `app/desktop/main.cjs`, `app/desktop/app.js`, `app/desktop/index.html`, `app/desktop/style.css`, `app/desktop/DESIGN-apple.md`, `app/desktop/package.json`.

---

## 4. Academic Papers & Project Customizations

- **LaTeX Manuscripts**:
  - `docs/main_paper/sn-article.tex` & `docs/main_pp_sci27/sn-article.tex`: *CRAM-ABSA: Multi-Task Rationale Alignment for Aspect-Based Hotel Review Assessment* (Introduction & RQ1–RQ3, Related Work, Proposed Methodology Stages 1–2, Experimental Evaluation, Discussion & Conclusion), compiled via `build.sh` with `sn-bibliography.bib` and 30 PDF references in `docs/src_paper/`.
- **Agent & Graphify Configuration**:
  - `.agents/rules/graphify.md`: Mandatory Graphify-first query & post-edit update protocol for coordinator and all swarm agents.
  - `.agents/skills/graphify/SKILL.md`: `/graphify` workflow skill for rebuilding, querying, and syncing `graphify-out/`.
  - `.agents/skills/absa-swarm/SKILL.md`: `/swarm` 6-specialist agent topology (`data-engineer`, `ml-trainer`, `benchmark-engineer`, `app-engineer`, `paper-docs`, `qa-verifier`, plus `monitor`).
