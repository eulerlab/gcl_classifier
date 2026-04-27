# GCL Classifier

A ganglion cell layer (GCL) classifier trained on functional two-photon calcium imaging recordings of mouse retinas in response to chirp and moving bar stimuli.
The labels and classification are based on Baden, Franke, Berens et al. (2016) "The functional diversity of retinal ganglion cells in the mouse." Nature 529.7586 (2016): 345-350. 
The classifier was already used in four publictions: 
1. Qiu, Y. et al. "Efficient coding of natural scenes improves neural system identification." PLoS computational biology 19.4 (2023): e1011037.
2. Gonschorek, D. et al. "Nitric oxide modulates contrast suppression in a subset of mouse retinal ganglion cells." Elife 13 (2025): RP98742.
3. Dyszkant, N. et al. "Photoreceptor degeneration has heterogeneous effects on functional retinal ganglion cell types." The Journal of Physiology (2025):, 603(21), 6599-6621.
4. Gonschorek, D. et al. "A large-scale dataset of functional mouse ganglion cell layer responses." bioRxiv (2025): 2025-12.


# Installation

## Quick Start
```bash
pip install gcl_classifier
```

If you additonally want to run the attached notebooks, install the extra dependencies using:
```bash
pip install "gcl_classifier[notebook]"
```

## First Time Setup

On first use, the model will be downloaded from Hugging Face Hub.
This happens automatically and only needs to be done once.
```python
from gcl_classifier import get_model

# Downloads model to ~/.cache/gcl_classifier/
model = get_model()
```

Let's use this model for celltype classification:
```python
import numpy as np
from gcl_classifier.data import get_data
from gcl_classifier.classifier import extract_features

# "Fake" preprocessed data for two cells
bar_ds_pvalues = np.array([0.04, 0.20])
roi_sizes_um2 = np.array([43.0, 56.5])
chirp_traces = np.random.random((2, 249))
bar_traces = np.random.random((2, 32))

# Load feat matrix to transform data into feature space
data = get_data()
chirp_features = data["chirp_feats"]
bar_features = data["bar_feats"]

# Extract the features for classifier
X, feature_names = extract_features(
    preproc_chirps=chirp_traces,
    preproc_bars=bar_traces,
    bar_ds_pvalues=bar_ds_pvalues,
    roi_size_um2s=roi_sizes_um2,
    chirp_features=chirp_features,
    bar_features=bar_features,
)

# Get predictions and probabilities
predictions = model.predict(X)
predictions_probs = model.predict_proba(X)
```

## Cache Location

Models (and training data) are cached at:
- Linux/Mac: `~/.cache/gcl_classifier/`
- Windows: `C:\Users\<username>\.cache\gcl_classifier\`

To clear cache and re-download the model:
```python
from gcl_classifier.model import load_model
model, model_dict = load_model(force_download=True)
```

You can do the same for the training data, but you only need to do this if you want to re-train the model.
```python
from gcl_classifier.data import load_data
data = load_data(force_download=True)
```

## Offline Usage

After initial download, the package works offline using the cached model.
