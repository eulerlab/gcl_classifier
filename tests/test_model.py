from gcl_classifier.model import load_model


class TestModelLoader:

    def test_model_download(self):
        """Test that model downloads without errors."""
        model = load_model(force_download=True)
        assert model is not None

    def test_model_load(self):
        """Test that model loads without errors."""
        model = load_model(force_download=False)
        assert model is not None

    def test_model_type(self):
        """Test that loaded model is correct type."""
        from sklearn.calibration import CalibratedClassifierCV
        model = load_model(force_download=False)
        assert isinstance(model, CalibratedClassifierCV)

    def test_model_has_predict_methods(self):
        """Test that model has required methods."""
        model = load_model(force_download=False)
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
        assert callable(model.predict)
        assert callable(model.predict_proba)
