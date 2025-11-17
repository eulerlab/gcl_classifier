# GCL Classifier

A ganglion cell layer (GCL) classifier trained on functional two-photon calcium imaging recordings of mouse retinas in response to chirp and moving bar stimuli.


# Installation

## Quick Start
```bash
pip install gcl_classifier
```

## First Time Setup

On first use, the model (~4GB) will be downloaded from Hugging Face Hub.
This happens automatically and only needs to be done once.
```python
from gcl_classifier import get_model

# Downloads model to ~/.cache/gcl_classifier/
model = get_model()
```

## Cache Location

Models are cached at:
- Linux/Mac: `~/.cache/gcl_classifier/`
- Windows: `C:\Users\<username>\.cache\gcl_classifier\`

To clear cache and re-download:
```python
from gcl_classifier.model import load_model
model = load_model(force_download=True)
```

## Offline Usage

After initial download, the package works offline using the cached model.