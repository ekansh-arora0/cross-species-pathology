# Model Checkpoints

This directory should contain the model weights for running experiments.

## Required Checkpoints

### 1. CPath-CLIP Vision Encoder
- **File**: `cpath_clip.pt`
- **Architecture**: ViT-L/14-336
- **Embedding dimension**: 3328
- **Source**: Contact authors for academic use
- **SHA256**: `[to be added upon release]`

### 2. Semantic Anchoring Projection Head (CLIP)
- **File**: `semantic_anchoring_clip_projection.pt`
- **Input dimension**: 768 (CLIP text encoder)
- **Output dimension**: 3328 (CPath-CLIP visual space)
- **Training**: Contrastive learning on TCGA-BRCA + CATCH
- **Download**: [To be hosted on HuggingFace/Zenodo]
- **SHA256**: `[to be added upon release]`

### 3. Semantic Anchoring Projection Head (Qwen)
- **File**: `semantic_anchoring_qwen_projection.pt`
- **Input dimension**: 1536 (Qwen2-1.5B)
- **Output dimension**: 3328
- **Download**: [To be hosted on HuggingFace/Zenodo]
- **SHA256**: `[to be added upon release]`

## Pre-computed Embeddings (Recommended)

For faster reproduction, download pre-computed embeddings:

```bash
# Download from Zenodo (DOI to be assigned)
wget https://zenodo.org/record/XXXXX/files/cpath_omni_embeddings.tar.gz
tar -xzf cpath_omni_embeddings.tar.gz -C data/embeddings/
```

Contents:
- `catch_train_embeddings.npy` - CATCH canine breast (train split, 2048 patches)
- `catch_train_labels.npy` - Corresponding labels
- `catch_test_embeddings.npy` - CATCH canine breast (test split, 20191 patches)
- `catch_test_labels.npy` - Corresponding labels
- `mast_cell_embeddings.npy` - Mast cell tumor (5530 patches)
- `mast_cell_labels.npy` - Corresponding labels
- `tcga_brca_embeddings.npy` - TCGA human breast (245000 patches)
- `tcga_brca_labels.npy` - Corresponding labels (all tumor)

## Reproducing Results

### With Pre-computed Embeddings (Fast)
```bash
# Download embeddings
mkdir -p data/embeddings
# [download command - see above]

# Reproduce Table 3
python scripts/reproduce_table3.py \
    --embeddings-dir data/embeddings \
    --projection-head checkpoints/semantic_anchoring_clip_projection.pt \
    --output-dir results/
```

### From Scratch (Requires GPUs)
```bash
# 1. Extract embeddings
python scripts/extract_embeddings.py \
    --data-dir data/patches \
    --checkpoint checkpoints/cpath_clip.pt \
    --output-dir data/embeddings

# 2. Train projection head
python scripts/train_projection.py \
    --embeddings-dir data/embeddings \
    --output checkpoints/my_projection.pt

# 3. Reproduce Table 3
python scripts/reproduce_table3.py \
    --embeddings-dir data/embeddings \
    --projection-head checkpoints/my_projection.pt
```

## Alternative Models

If CPath-CLIP weights are not available, you can use:

### OpenAI CLIP
```python
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai')
```

### H-optimus-0
```python
import timm
model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True)
```
- Embedding dimension: 1536
- No text anchoring needed (already achieves 79.63% cross-species AUC)

### Phikon-v2
```python
import timm
model = timm.create_model("hf-hub:owkin/phikon-v2", pretrained=True)
```

## Directory Structure

```
checkpoints/
├── README.md           # This file
├── cpath_clip.pt       # Vision encoder (required)
├── text_projection.pt  # CLIP text projection (optional)
└── qwen_projection.pt  # Qwen text projection (optional)
```

## Training Your Own

To train a text projection head on your own data:

```python
from models import SemanticAnchoring, SemanticAnchoringTrainer
from models.text_encoder import QwenTextEncoder

# Initialize
text_encoder = QwenTextEncoder()
model = SemanticAnchoring(text_encoder)

# Set prompts
model.set_anchors({
    "tumor": "Malignant tumor tissue with abnormal cellular proliferation",
    "normal": "Normal healthy tissue with regular architecture"
})

# Train
trainer = SemanticAnchoringTrainer(model)
trainer.train(train_embeddings, train_labels, epochs=100)

# Save
torch.save(model.projection.state_dict(), "checkpoints/my_projection.pt")
```

## Downloading Pre-computed Embeddings

For reproducing paper results without running inference:

```bash
# Download pre-computed embeddings (if available)
wget https://example.com/cpath_omni_embeddings.tar.gz
tar -xzf cpath_omni_embeddings.tar.gz -C data/
```

Files included:
- `tcga_brca_embeddings.npy` - Human breast cancer (TCGA)
- `catch_breast_embeddings.npy` - Canine breast cancer (CATCH)
- `mast_cell_embeddings.npy` - Canine mast cell tumor
- `*_labels.npy` - Corresponding labels
