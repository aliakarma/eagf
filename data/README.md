# Data

This directory holds the datasets used by EAGF across both case studies.
No data files are committed to this repository. Follow the instructions below
to obtain and prepare each dataset.

---

## Case Study 1: Biometric Security (EFR Dataset)

### Source

| Field | Value |
|---|---|
| Dataset name | Employee Face Recognition (EFR) |
| Source | Kaggle community upload |
| URL | https://www.kaggle.com/datasets/smmmmmmmmmmmm/employee-face-recognition |
| Size | ~13,000 images |
| Format | JPEG, organised by subject subdirectory |

### Important Limitation

> **The EFR dataset is a Kaggle community upload without formal demographic
> certification.** Demographic labels (gender, skin tone) have not been validated
> by an independent annotator and may contain label noise that affects RP and FPRP
> measurements. For production or regulatory purposes, use an institutionally
> certified benchmark:
>
> | Certified Benchmark | URL |
> |---|---|
> | CelebA | https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html |
> | VGGFace2 | https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/ |
> | DemogPairs | https://ihupont.github.io/publications/2019-demogpairs |

### Download Instructions

```bash
# Option A: Kaggle API (requires ~/.kaggle/kaggle.json)
pip install kaggle
kaggle datasets download smmmmmmmmmmmm/employee-face-recognition \
    -p data/biometric/ --unzip

# Option B: Manual download
# 1. Visit https://www.kaggle.com/datasets/smmmmmmmmmmmm/employee-face-recognition
# 2. Click Download, extract to data/biometric/efr_raw/
```

### Preprocessing

After download, run:

```bash
python src/utils/preprocessing.py \
    --input  data/biometric/efr_raw/ \
    --output data/biometric/efr_processed/ \
    --remove-single-sample-classes \
    --target-size 10000 \
    --demographic-labels gender skin_tone \
    --seed 42
```

This script:
1. Removes classes with only one image (mitigates class-imbalance confounds)
2. Assigns demographic pseudo-labels from filename metadata
3. Applies stratified train/val/test split (70/15/15) by demographic group
4. Saves split indices to `data/biometric/splits.json` for reproducibility

### Expected directory structure after preprocessing

```
data/biometric/
├── efr_raw/                 ← Raw downloaded images
├── efr_processed/
│   ├── train/               ← 7,015 images (70%)
│   ├── val/                 ← 1,503 images (15%)
│   └── test/                ← 1,503 images (15%)
├── splits.json              ← Deterministic split indices (seed=42)
└── demographics.csv         ← Per-image demographic labels
```

---

## Case Study 2: RE-IoT Anomaly Detection (Synthetic Telemetry)

No external download is required. Synthetic 5G RE-IoT telemetry is generated
locally using the EAGF simulation module, parameterised from published microgrid
operational data (Liang et al., 2017, IEEE Trans. Smart Grid).

### Generation

```bash
python -m src.utils.data_loader \
    --dataset      reiot \
    --output       data/reiot/ \
    --nodes        120 \
    --urban        40 \
    --periurban    40 \
    --rural        40 \
    --attack-ratio 0.05 \
    --attacks      fdia command_injection dos \
    --frequency    1.0 \
    --duration     86400 \
    --seed         42
```

### Node Profile Parameters (from Liang et al., 2017)

| Node Class | Load Variation | Attack Base Rate | Count |
|---|:---:|:---:|:---:|
| Urban (grid-tied) | ±2% | 5% | 40 |
| Peri-urban | ±8% | 5% | 40 |
| Rural (off-grid) | ±22% | 5% | 40 |

### Expected directory structure after generation

```
data/reiot/
├── train/                   ← 96 nodes (80%)
│   ├── urban/
│   ├── periurban/
│   └── rural/
├── test/                    ← 24 nodes (20%, stratified)
│   ├── urban/
│   ├── periurban/
│   └── rural/
├── telemetry_metadata.json  ← Node profiles and attack injection log
└── generation_config.yaml   ← Full reproducibility record
```

---

## Data Availability Statement

In accordance with journal data availability policies:

- The EFR biometric dataset is publicly available at the Kaggle URL above.
- The synthetic RE-IoT telemetry is fully reproducible from the generation script
  above with `--seed 42`. The generation script and all parameterisation details
  are included in this repository.
- Pre-processed split indices (`splits.json`) are committed to this repository
  to guarantee identical train/val/test partitions across all experiments.
- No personally identifiable information beyond the EFR dataset is used.
  Smart-meter consumption data is synthetic only.
