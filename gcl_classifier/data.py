from pathlib import Path

import numpy as np

from gcl_classifier.labels import baden_cluster_id_to_group_id, baden_group_id_to_supergroup

HF_REPO_ID = "eulerlab/gcl_classifier_training_data"
MAT_FILENAMES = ("RGCData_postprocessed.mat", "bar_feats.npz", "chirp_feats.npz")


def load_data(force_download: bool = False):
    """
    Load the training data.

    On first use, downloads data from Hugging Face Hub.
    Subsequent calls use cached version.

    Args:
        force_download: If True, re-download even if cached

    Returns:
        MatLab file
    """
    from huggingface_hub import hf_hub_download

    try:
        cache_paths = [hf_hub_download(
            repo_type='dataset',
            repo_id=HF_REPO_ID,
            filename=f,
            cache_dir=Path.home() / ".cache" / "gcl_classifier",
            force_download=force_download,
        ) for f in MAT_FILENAMES]

        baden_mat = load_baden_mat(cache_paths[0])
        bar_feats = np.load(cache_paths[1])
        chirp_feats = np.load(cache_paths[2])

        data = {
            "baden_mat": baden_mat,
            "bar_feats": bar_feats,
            "chirp_feats": chirp_feats,
        }

        return data

    except Exception as e:
        raise RuntimeError(
            f"Failed to load data from Hugging Face Hub. "
            f"Please check your internet connection and try again. "
            f"Error: {e}"
        )


# Lazy loading - only download when first accessed
_data = None


def get_data() -> dict:
    """Get data instance, downloading on first call."""
    global _data
    if _data is None:
        print("Loading data (downloading if needed)...")
        _data = load_data()
        print("✓ Data loaded successfully")
    return _data


def load_baden_mat(filename) -> dict:
    from scipy.io import loadmat
    baden_mat = loadmat(
        filename,
        struct_as_record=True, matlab_compatible=False, squeeze_me=True, simplify_cells=True
    )['data']
    return baden_mat


def prepare_baden_data(baden_mat, quality_filter: bool = True) -> dict:
    roi_size_um2 = baden_mat['info']['area2d']

    chirp_traces = baden_mat['chirp']['traces'].T
    chirp_qi = baden_mat['chirp']['qi']

    bar_traces = baden_mat['ds']['tc'].T
    bar_qi = baden_mat['ds']['qi']
    bar_dsi = baden_mat['ds']['dsi']
    bar_dp = baden_mat['ds']['dP']

    cluster_labels = np.asarray(baden_mat['info']['final_idx']).flatten()
    group_labels = np.array([baden_cluster_id_to_group_id(c_label) for c_label in cluster_labels])
    super_labels = np.array([baden_group_id_to_supergroup(g_label) for g_label in group_labels])

    if quality_filter:
        filt_idx = (cluster_labels > 0) & ((bar_qi > 0.6) | (chirp_qi > 0.45))
    else:
        filt_idx = np.ones(cluster_labels.size, dtype=bool)

    baden_data = {
        "cluster_labels": cluster_labels[filt_idx],
        "group_labels": group_labels[filt_idx],
        "super_labels": super_labels[filt_idx],
        "chirp_traces": chirp_traces[filt_idx],
        "chirp_qi": chirp_qi[filt_idx],
        "bar_traces": bar_traces[filt_idx],
        "bar_qi": bar_qi[filt_idx],
        "bar_dsi": bar_dsi[filt_idx],
        "bar_dp": bar_dp[filt_idx],
        "roi_size_um2": roi_size_um2[filt_idx],
    }
    return baden_data
