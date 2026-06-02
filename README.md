# PolySIM 2026 Challenge — Multimodal Speaker Identification

A multimodal speaker identification system for the PolySIM 2026 Challenge.  
The system identifies speakers from **audio + face** features in both **English (in-language)** and **Urdu (cross-lingual)** settings.

---

## Competition Protocol Compliance

This submission **fully complies** with the [PolySIM Evaluation Protocols](./PolySIM%20Evaluation%20Protocols.md):

| Protocol | Training Data | Test Data |
|----------|--------------|-----------|
| **P3** In-language multimodal | English Face + Voice ✅ | English Face + Voice ✅ |
| **P4** In-language missing-modality | English Face + Voice ✅ | English Voice-only ✅ |
| **P5** Cross-lingual multimodal | English Face + Voice ✅ | Urdu Face + Voice ✅ |
| **P6** Cross-lingual missing-modality | English Face + Voice ✅ | Urdu Voice-only ✅ |

---

## Project Structure

```
2026polysim/
├── main.py              # Training script
├── submit.py            # Inference + submission generation
├── config_all.py        # All configuration parameters
├── models/
│   ├── model.py         # Base components
│   └── multibranch.py   # main model
├── utils/
│   ├── featLoader.py    # Data loader
│   ├── trainer.py       # Multi-stage trainer
│   ├── losses.py        # OrthogonalProjectionLoss, CenterLoss
│   └── post_process.py  # Post-processing: prototype refinement, Graph LP
├── checkpoints/         # Saved model checkpoint
├── csv_files/           # Training CSV and submission output
├── test_set/            # Test dataset
│   ├── csv/comp/        # v1_test_English.csv, v1_test_Urdu.csv
│   └── feat/            # Test feature files
├── feats/               # Training feature files (English only, .npy)
├── log/                 # Training logs
└── README.md
```

---

## Environment Setup

### Requirements
- Python 3.10
- PyTorch 2.6 with CUDA 12.4
- Git LFS (for checkpoint download)

### Install

```bash
git lfs install

git clone git@github.com:2351548518/Polysim2026code.git

cd Polysim2026code

git lfs pull

conda create -n polysim python=3.10

conda activate polysim

pip install -r requirements.txt
```

---

## Training

Train the model from scratch using only English data:

```bash
python main.py
```

### Checkpoint

After training, the checkpoint is saved to:
```
checkpoints/v1_all_alpha0_multibranch_cross_attention_center_loss_all_missinglearning.pt
```

---

## Inference & Submission

### 1. Prepare Test Data

Ensure test data is placed in `test_set/`:
- `test_set/csv/comp/v1_test_English.csv`
- `test_set/csv/comp/v1_test_Urdu.csv`
- `test_set/feat/` — feature files (.npy)

### 2. Run Submission Script

```bash
python submit.py
```

### 3. Output

Generated files are written to `csv_files/submission_combined/`:

```text
csv_files/submission_combined/
├── submission_v1_test_English_English.csv   # P3, P4 predictions
└── submission_v1_test_English_Urdu.csv      # P5, P6 predictions
```

---

## Model Architecture

**MultiBranchFOP** (`models/multibranch.py`) features:
- **Three classification heads**: Face, Audio, and Fusion
- **Transformer-based embedding branches** with learnable tokens
- **Reliability-weighted fusion**: Learned per-modality reliability scores (sigmoid gates)
- **Learnable Missing Token**: Replaces all-zero face inputs for graceful audio-only inference
- **Cross-Attention fusion**: Face tokens attend to audio tokens and vice versa
