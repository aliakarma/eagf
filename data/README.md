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

## Case Study 2: IIoT Intrusion Detection (Edge-IIoTset)

### Source

| Field | Value |
|---|---|
| Dataset name | Edge-IIoTset |
| Source | IEEE Access 2022 (Ferrag et al.) |
| URL | https://dx.doi.org/10.21203/rs.3.rs-1433551/v1 |
| Size | ~78 MB (157,800 flow records) |
| Features | 40 network flow + protocol-specific attributes |
| Labels | Normal vs. Attack (binary, 1:5.8 imbalance) |
| Protected groups | Protocol type (web, IoT-MQTT, misc) |

### Download Instructions

1. Download `ML-EdgeIIoT-dataset.csv` from the URL above
2. Place at `data/real_iot/edge_iiot.csv`

### Expected directory structure

```
data/real_iot/
└── edge_iiot.csv            ← 157,800 rows, ~78 MB
```

The pipeline splits 70/15/15 (train/val/test) with stratification by label
and protocol-type group. Splitting is deterministic per seed.

---

## Data Availability Statement

- The EFR biometric dataset is publicly available at the Kaggle URL above.
- The Edge-IIoTset is publicly available from IEEE DataPort (Ferrag et al., 2022).
- Pre-processed split indices are deterministic from the seed parameter.
- No personally identifiable information is used. EFR face images are
  used for identity classification only.
