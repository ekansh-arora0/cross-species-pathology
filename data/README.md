# Data Directory Structure

This directory contains data split configurations and should be populated with pathology data for experiments.

## Train/Test Splits

The exact splits used in the paper are provided in two formats:

- **`splits.json`**: Comprehensive split file with all slide IDs and patch counts
- **`splits/splits_config.json`**: Detailed configuration with preprocessing parameters

### Key Split Information

**CATCH Canine Breast Cancer (21 slides)**:
- **Train (1 slide)**: `50cf88e9a33df0c0c8f9.svs` 
  - 2,048 patches (1,003 tumor, 1,045 normal)
  - Used for prototype computation in few-shot setting
- **Test (20 slides)**: All other slides
  - 20,191 patches (12,857 tumor, 7,334 normal)

**MITOS_WSI_CCMCT Mast Cell Tumor (7 annotated slides)**:
- Test only: `013.tiff`, `014.tiff`, `017.tiff`, `018.tiff`, `021.tiff`, `024.tiff`, `069.tiff`
- 5,530 patches (3,189 tumor, 2,341 normal)

**TCGA-BRCA Human Breast Cancer (1,098 slides)**:
- All diagnostic slides used as source domain
- ~245,000 patches extracted

## Data Sources

### 1. CATCH Canine Mammary Carcinoma Dataset
- **Source**: https://github.com/DeepPathology/CATCH
- **License**: CC BY-NC-SA 4.0
- **Format**: SVS whole slide images with SlideRunner XML annotations
- **Download**: Contact dataset maintainers

### 2. MITOS_WSI_CCMCT Canine Mast Cell Tumor
- **Source**: https://mitos-atypia-14.grand-challenge.org/
- **License**: Research use only
- **Format**: TIFF whole slide images
- **Annotations**: SlideRunner SQLite database (`MITOS_WSI_CMC_CODAEL_TR_ROI.sqlite`)

### 3. TCGA-BRCA Human Breast Cancer
- **Source**: https://portal.gdc.cancer.gov/
- **Project ID**: TCGA-BRCA
- **Data Type**: Diagnostic Slide Images
- **Download**: Use GDC Data Transfer Tool
  ```bash
  # Install GDC client
  pip install gdc-client
  
  # Download manifest from GDC portal, then:
  gdc-client download -m manifest.txt
  ```

## Expected Directory Structure

After downloading and preprocessing, organize data as:

```
data/
├── splits.json                    # Train/test split definitions
├── splits/
│   └── splits_config.json        # Detailed configuration
│
├── catch_canine_breast/          # CATCH dataset
│   ├── slides/
│   │   ├── 50cf88e9a33df0c0c8f9.svs    # Train slide
│   │   ├── 022857018aa597374b6c.svs    # Test slides
│   │   └── ...
│   ├── annotations/
│   │   └── MITOS_WSI_CMC_CODAEL_TR_ROI.sqlite
│   └── patches/                   # Extracted patches (optional)
│       ├── train/
│       │   ├── tumor/
│       │   └── normal/
│       └── test/
│           ├── tumor/
│           └── normal/
│
├── mitos_mast_cell/              # Mast cell tumor dataset
│   ├── slides/
│   │   ├── 013.tiff
│   │   └── ...
│   └── patches/
│
├── tcga_brca/                    # Human breast cancer
│   ├── slides/
│   │   ├── TCGA-XX-XXXX-01Z-00-DX1.svs
│   │   └── ...
│   └── patches/
│
└── embeddings/                   # Pre-computed embeddings (recommended)
    ├── catch_train_embeddings.npy
    ├── catch_train_labels.npy
    ├── catch_test_embeddings.npy
    ├── catch_test_labels.npy
    ├── mast_cell_embeddings.npy
    ├── mast_cell_labels.npy
    ├── tcga_brca_embeddings.npy
    └── tcga_brca_labels.npy
```

## Preprocessing Pipeline

Use the preprocessing scripts to extract patches from WSIs:

```bash
# 1. Extract patches from CATCH canine breast slides
python preprocessing/patch_extraction.py \
    --input-dir data/catch_canine_breast/slides \
    --output-dir data/catch_canine_breast/patches \
    --annotations data/catch_canine_breast/annotations/MITOS_WSI_CMC_CODAEL_TR_ROI.sqlite \
    --patch-size 1024 \
    --stride 2048 \
    --normalize macenko

# 2. Extract embeddings using CPath-CLIP
python scripts/extract_embeddings.py \
    --data-dir data/catch_canine_breast/patches \
    --checkpoint checkpoints/cpath_clip.pt \
    --output data/embeddings/catch_embeddings.npy
```

## Using Pre-computed Embeddings

For faster reproduction, download pre-computed embeddings:

```bash
# Download from Zenodo (DOI will be assigned upon release)
wget https://zenodo.org/record/XXXXX/files/cpath_omni_embeddings.tar.gz
tar -xzf cpath_omni_embeddings.tar.gz -C data/embeddings/
```

## Verifying Data Integrity

After setup, verify your data matches the expected counts:

```python
import numpy as np
import json

# Load splits
with open('data/splits.json') as f:
    splits = json.load(f)

# Verify CATCH counts
catch = splits['catch_canine_breast']
print(f"CATCH train patches: {catch['train_patches']['total']} (expected: 2048)")
print(f"CATCH test patches: {catch['test_patches']['total']} (expected: 20191)")

# If using pre-computed embeddings
train_emb = np.load('data/embeddings/catch_train_embeddings.npy')
test_emb = np.load('data/embeddings/catch_test_embeddings.npy')
print(f"Train embeddings shape: {train_emb.shape}")
print(f"Test embeddings shape: {test_emb.shape}")
```

Before running experiments, preprocess your data:

```bash
# For WSI data
python preprocessing/patch_extraction.py \
    --input data/tcga_brca/slides \
    --output data/tcga_brca/patches \
    --patch-size 1024 \
    --stride 512

# Stain normalization (optional, done on-the-fly during inference)
python preprocessing/macenko_normalizer.py \
    --input data/dog_breast \
    --output data/dog_breast_normalized
```

## Data Format

### Patch Images
- Format: PNG or JPEG
- Size: 224x224 to 1024x1024 (resized during inference)
- Color: RGB (H&E stained)

### Labels
- Directory-based: Place in `tumor/` or `normal/` subdirectories
- CSV-based: Create `labels.csv` with columns `filename,label`

### WSI Support
- Formats: SVS, NDPI, TIFF (OpenSlide compatible)
- Automatic tissue detection and patch extraction
