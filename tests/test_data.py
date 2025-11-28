import pytest
import numpy as np

from gcl_classifier.data import load_data, prepare_baden_data, get_data


class TestDataLoader:


    @pytest.mark.parametrize("force_download", [True, False])
    def test_data_load(self, force_download: bool):
        """Test that data downloads without errors with both force_download options."""
        data = load_data(force_download=force_download)
        assert data is not None
        assert isinstance(data, dict)
        assert 'baden_mat' in data.keys()
        assert 'bar_feats' in data.keys()
        assert 'chirp_feats' in data.keys()
        assert isinstance(data['baden_mat'], dict)
        assert isinstance(data['bar_feats'], np.ndarray)
        assert isinstance(data['chirp_feats'], np.ndarray)

    def test_extract_baden_data(self):
        """Test that data extracts without errors."""
        data = get_data()
        baden_data = prepare_baden_data(data['baden_mat'])
        assert baden_data is not None
        assert isinstance(baden_data, dict)
        assert 'cluster_labels' in baden_data.keys()
