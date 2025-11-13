import pickle

import numpy as np

from rgc_classifier.baden16 import baden_cluster_id_to_group_id, baden_group_id_to_supergroup, BADEN_CLUSTER_INFO


def classify_cells(preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s,
                   chirp_features, bar_features, classifier):
    features, feature_names = extract_features(
        preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s, chirp_features, bar_features)
    probs = classifier.predict_proba(features)
    return probs


def baden16_cluster_probs_to_info(probs):
    if len(probs) != 75:
        raise ValueError(f"Expected 75 probabilities corresponding to 75 Baden clusters, got {len(probs)}.")

    cluster_id = np.argmax(probs) + 1  # Cluster IDs are 1-indexed
    group_id = baden_cluster_id_to_group_id(cluster_id)
    supergroup = baden_group_id_to_supergroup(group_id)
    prob_cluster = probs[cluster_id - 1]

    group_ids = BADEN_CLUSTER_INFO[:, 2].astype(int)
    supergroups = BADEN_CLUSTER_INFO[:, 3].astype(str)

    prob_group = np.sum(probs[group_ids == group_id])
    prob_supergroup = np.sum(probs[supergroups == supergroup])
    prob_rgc = np.sum(probs[supergroups != 'dAC'])
    prob_class = (1. - prob_rgc) if supergroup == 'dAC' else prob_rgc

    return cluster_id, group_id, supergroup, prob_cluster, prob_group, prob_supergroup, prob_class


def extract_features(preproc_chirps, preproc_bars, bar_ds_pvalues, roi_size_um2s, chirp_features, bar_features):
    features = np.concatenate([
        np.dot(preproc_chirps, chirp_features),
        np.dot(preproc_bars, bar_features),
        bar_ds_pvalues[:, np.newaxis],
        roi_size_um2s[:, np.newaxis]
    ], axis=-1)

    feature_names = [f'chirp_{i}' for i in range(chirp_features.shape[1])] + \
                    [f'bar_{i}' for i in range(bar_features.shape[1])] + ['bar_ds_pvalue', 'roi_size_um2']

    return features, feature_names


def check_classifier_dict(clf_dict):
    assert type(clf_dict) == dict, "Classifier file must contain a dictionary with classifier data."

    # Check keys
    assert 'classifier' in clf_dict, "Classifier dictionary must contain a 'classifier' key."
    assert 'chirp_feats' in clf_dict, "Classifier dictionary must contain a 'chirp_feats' key."
    assert 'bar_feats' in clf_dict, "Classifier dictionary must contain a 'bar_feats' key."
    assert 'feature_names' in clf_dict, "Classifier dictionary must contain a 'feature_names' key."
    assert 'train_x' in clf_dict, "Classifier dictionary must contain a 'train_x' key."
    assert 'train_y' in clf_dict, "Classifier dictionary must contain a 'train_y' key."
    assert 'y_names' in clf_dict, "Classifier dictionary must contain a 'y_names' key."

    # Chek value
    assert isinstance(clf_dict['train_x'], np.ndarray), "The 'train_x' key must contain a numpy array."
    assert isinstance(clf_dict['train_y'], np.ndarray), "The 'train_y' key must contain a numpy array."
    assert clf_dict['train_x'].shape[0] == clf_dict[
        'train_y'].size, "The number of samples in 'train_x' and 'train_y' must match."

    for val in np.unique(clf_dict['train_y']):
        assert val in clf_dict['y_names'].keys(), f"Value {val} in 'train_y' not found in 'y_names'."

    # Check if classifier is a valid scikit-learn classifier
    from sklearn.base import is_classifier
    assert is_classifier(clf_dict['classifier']), "The 'classifier' key must contain a valid scikit-learn classifier."

    return clf_dict


def save_classifier_to_file(classifier, chirp_feats, bar_feats, feature_names, train_x, train_y, y_names,
                            classifier_file, **kwargs):
    """
    Saves the classifier and its metadata to a file.
    """
    clf_dict = {
        'classifier': classifier,
        'chirp_feats': chirp_feats,
        'bar_feats': bar_feats,
        'feature_names': feature_names,
        'train_x': train_x,
        'train_y': train_y,
        'y_names': y_names,
        **kwargs
    }

    check_classifier_dict(clf_dict)

    with open(classifier_file, 'wb') as f:
        pickle.dump(clf_dict, f)
