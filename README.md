# Satellite Image Land-Use Classifier & Temporal Change Detector

A computer vision system that classifies land-use types from satellite imagery and detects land-cover changes between two simulated time periods, using transfer learning and embedding-based change detection.

## Project Structure

```
.
├── notebooks/
│   ├── 01-data-pipeline.ipynb        # EuroSAT download, spatial block split, class distribution
│   ├── 02-baseline-cnn.ipynb         # Scratch 3-layer CNN baseline
│   ├── 03-transfer-learning.ipynb    # Two-phase fine-tuning + UC Merced holdout + GradCAM
│   ├── 04-change-detection.ipynb     # Embeddings, cosine similarity, ROC, GradCAM, t-SNE comparison
│   └── 05-evaluation.ipynb           # Spatial leakage experiment, error analysis, imbalance experiment
├── app/
│   └── dashboard.py                  # Streamlit geo-dashboard with multi-threshold toggle
├── checkpoints/
│   ├── finetuned_resnet18_day4.pt    # Fine-tuned ResNet-18 used by the dashboard
│   └── thresholds.json               # High recall / balanced / high precision thresholds
├── test_images/                      # Sample EuroSAT tiles for testing the dashboard
├── reports/
│   └── project_report.pdf
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate       # Mac/Linux

pip install -r requirements.txt
```

## Datasets

- **EuroSAT** — 27,000 satellite tiles, 10 land-use classes (Sentinel-2, 10m/pixel). Used for training/validation. [Kaggle: apollo2506/eurosat-dataset]
- **UC Merced Land Use** — 2,100 images, 21 classes (USGS aerial imagery, ~0.3m/pixel). Used as an out-of-domain holdout set. [Kaggle: abdulhasibuddin/uc-merced-land-use-dataset]

## Methodology Notes

- **Spatial block split, not random split.** EuroSAT tiles are grouped into pseudo-spatial "blocks" (same-class images grouped by filename ID) and whole blocks are assigned to train/val/test, to avoid leakage from near-duplicate tiles landing on both sides of a split. A naive random split is also built and used *only* for the leakage comparison in `05-evaluation.ipynb` — never for real training.
- **T1/T2 change detection is simulated, not real temporal data.** EuroSAT has no timestamps. "Before/after" pairs are constructed by splitting each spatial block in half, with a random subset of "after" images deliberately swapped to a different class to create labeled ground-truth "changed" pairs.
- **UC Merced evaluation uses a partial class mapping.** Only 8 of EuroSAT's 10 classes have a reasonable conceptual match in UC Merced's 21 classes (PermanentCrop and SeaLake have no match and are excluded). This is a real limitation, stated explicitly rather than implying full 21-class coverage.

## Modules

### Module 1 — Land-Use Classifier
Transfer learning (ResNet-18, ImageNet pretrained) with a two-phase fine-tuning strategy:
- **Phase 1** — freeze backbone, train classifier head only, 3 epochs
- **Phase 2** — unfreeze last 2 conv blocks (`layer3`, `layer4`), reduce LR 10×, train 5 more epochs

### Module 2 — Temporal Change Detector
Reuses Module 1's backbone as a frozen feature extractor (classifier head replaced with identity). Extracts 512-dim embeddings, simulates T1/T2 pairs via block partitioning, flags changes via cosine similarity against a threshold selected using Youden's J statistic on the ROC curve.

### Module 3 — Geo-Dashboard
Streamlit app: upload two tiles (before/after), get predicted land-use class + confidence for each, cosine similarity score, pixel-difference heatmap, and a change flag — with a sensitivity toggle (Bonus B) to switch between high recall / balanced / high precision operating points.

## Running the Dashboard

```bash
streamlit run app/dashboard.py
```
Opens at `http://localhost:8501`. Sample test tiles are in `test_images/` if you want to try it without your own images.

## Results Summary

| Model | EuroSAT Val Macro-F1 | EuroSAT Val Accuracy |
|---|---|---|
| Baseline CNN (scratch, 3-layer) | 0.779 | 78.3% |
| Fine-tuned ResNet-18 (two-phase) | 0.956 | 95.7% |

| Evaluation | Accuracy |
|---|---|
| Fine-tuned model — EuroSAT val (in-domain) | 95.7% |
| Fine-tuned model — UC Merced holdout (cross-domain, 8/10 classes) | 29.7% |

The large in-domain vs. cross-domain gap reflects a genuine sensor/resolution mismatch (Sentinel-2 satellite vs. USGS aerial imagery) rather than a modeling error — discussed in the project report.

### Change Detection
- **ROC AUC:** 0.976
- **Selected threshold (balanced, Youden's J):** 0.443 — TPR 0.931, FPR 0.094
- Three operating points available in the dashboard toggle: high recall (0.486), balanced (0.443), high precision (0.369)

### Spatial Leakage Experiment
| Split Strategy | Val Accuracy |
|---|---|
| Random split (naive) | 85.7% |
| Spatial block split (correct) | 78.3% |

A 7.4-point gap from split strategy alone, on an identical model — demonstrates real data leakage from naive random splitting on spatially correlated tiles. Full write-up in `05-evaluation.ipynb` and the project report.

## Bonus Tasks Implemented

- [x] **A — GradCAM visualisation.** Implemented on the fine-tuned ResNet-18 (`layer4`), 3 interpreted examples. See `04-change-detection.ipynb`.
- [x] **B — Multi-threshold toggle.** Dashboard includes a high recall / balanced / high precision sensitivity switch, backed by real ROC-derived thresholds.
- [x] **C — Embedding visualisation.** t-SNE projection of scratch CNN vs. fine-tuned ResNet-18 embeddings, side by side, same 3,000 sampled images. See `04-change-detection.ipynb`.
- [x] **D — Imbalance experiment.** Highway and PermanentCrop downsampled to 20%; macro-F1 dropped from 0.779 → 0.687. Weighted cross-entropy loss (class weights ~4.6-4.7× for the two downsampled classes) recovered macro-F1 to 0.810. Full comparison table and write-up in `05-evaluation.ipynb`.

## Error Analysis

Top-5 highest-confidence misclassifications (all >99.99% confidence, all wrong) analyzed in `05-evaluation.ipynb`. Common pattern: every error occurs between conceptually adjacent classes (e.g. AnnualCrop↔Pasture, River↔Highway, PermanentCrop↔AnnualCrop) — the model fails at fine-grained distinctions between visually similar land-use types, not at random.

## Report & Demo

- https://github.com/vaibhavpaliwal00-ui/satellite-land-use-classifier/blob/main/project_report.pdf
- https://youtu.be/UKI7ZwEbCBM?si=fiAHpNu8qtgIOt5y
