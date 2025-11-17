from pathlib import Path

import skops.io as sio

HF_REPO_ID = "eulerlab/gcl_classifier"
MODEL_FILENAME = "gcl_classifier.skops"


def load_model(force_download=False):
    """
    Load the pre-trained calibrated classifier.

    On first use, downloads model from Hugging Face Hub.
    Subsequent calls use cached version.

    Args:
        force_download: If True, re-download even if cached

    Returns:
        Trained CalibratedClassifierCV model
    """
    from huggingface_hub import hf_hub_download

    try:
        # Download (or get from cache)
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME,
            cache_dir=Path.home() / ".cache" / "gcl_classifier",
            force_download=force_download,
        )

        # Load with skops
        model = sio.load(
            model_path,
            trusted=[
                'sklearn.calibration._CalibratedClassifier',
                'sklearn.calibration._SigmoidCalibration'
            ])

        return model

    except Exception as e:
        raise RuntimeError(
            f"Failed to load model from Hugging Face Hub. "
            f"Please check your internet connection and try again. "
            f"Error: {e}"
        )


# Lazy loading - only download when first accessed
_model = None


def get_model():
    """Get model instance, downloading on first call."""
    global _model
    if _model is None:
        print("Loading model (downloading if needed)...")
        _model = load_model()
        print("✓ Model loaded successfully")
    return _model


# For convenience, expose at module level
# This won't download until actually used
def predict(X):
    """Make predictions using the model."""
    return get_model().predict(X)


def predict_proba(X):
    """Get prediction probabilities."""
    return get_model().predict_proba(X)
