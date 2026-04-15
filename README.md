# Seeing Surgical Skill

**Vision-Based Hand Motion Analysis for Surgical Skill Assessment**

This repository accompanies the master thesis:

> *Seeing Surgical Skill: Vision-Based Hand Motion Analysis for Assessment in Orthopedic Trauma Training*
> 
> Finn Weikert, 2026

---

## 🔍 Overview

This project explores whether **markerless hand tracking from video** can be used to assess surgical skill in orthopedic training.

**Pipeline:**

1. Extract hand landmarks from video (MediaPipe)
2. Build kinematic motion representations
3. Model surgical skill using:

   * Global motion efficiency (PC1)
   * Local motion structure (BoW, MIL)

**Key takeaway:**
A single global efficiency component explains most of the skill signal, while local motion patterns provide consistent improvements.

---

## 📁 Repository Structure

```
├── data/
├── notebooks/
├── scripts/
├── src/thesis_package/
├── results/
├── saved_models/
├── early_experiments/
├── thesis.pdf
└── environment.yml
```

### `data/`

* Pre-extracted landmark trajectories (`.pkl`)
* Processed palm trajectories + orientation landmarks
* Expert skill scores (`.csv`)
* Example videos (raw + tracking overlay)
* `metrics/pc1_features.csv` (precomputed global features)

> ⚠️ Original surgical videos are **not included** (confidential)

---

### `notebooks/` (main entry point)

All experiments, figures, and results from the thesis:

* `01_exploration_tracking.ipynb` – tracking & preprocessing
* `02_global_feature_screening.ipynb` – global features + PC1
* `03_window_feature_set_evaluation.ipynb` – local features
* `04_bow_analysis.ipynb` – Bag-of-Words modeling
* `05_mil_attention_analysis.ipynb` – MIL + attention
* `Exploratory_motion_ae.ipynb` – autoencoder experiments

---

### `scripts/` (reproduce results)

Run the main experiments without notebooks:

```bash
python scripts/run_global_models.py
python scripts/run_bow_models.py
python scripts/run_mil_models.py
```

---

### `src/thesis_package/`

Core implementation:

* `features/` – feature extraction (global + local + BoW)
* `models/` – MLP, MIL (attention), autoencoder
* `tracking/` – MediaPipe + postprocessing
* `training/` – LOSO evaluation + training loops

---

### `results/`

* `figures/` – thesis figures
* `tables/` – reproduced performance tables

---

### `saved_models/`

Pretrained MIL models (LOSO × ensemble).
➡️ Included to **avoid long retraining times**

---

### `early_experiments/`

Unused / exploratory code (not maintained)

---

## ⚙️ Setup

```bash
conda env create -f environment.yml
conda activate env-ml
```

Launch notebooks:

```bash
jupyter notebook
```

---

## 🚀 Reproducibility

* Use scripts for clean reproduction
* Use `saved_models/` to skip heavy training
* All reported results are reproducible from this repo

---

## 📌 Notes

* Data is partially shared (no raw videos)
* Focus is on **methodology + modeling**, not dataset release
* All figures/tables in the thesis are generated from this code

---

## 📄 Thesis

See `thesis.pdf` for full details:

* methodology
* experiments
* discussion

---

## 📬 Contact

For questions, feel free to reach out.

