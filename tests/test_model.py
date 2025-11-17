import numpy as np

from gcl_classifier.model import load_model


class TestModelLoader:

    def test_model_loads(self):
        """Test that model loads without errors."""
        model = load_model()
        assert model is not None

    def test_model_type(self):
        """Test that loaded model is correct type."""
        from sklearn.calibration import CalibratedClassifierCV
        model = load_model()
        assert isinstance(model, CalibratedClassifierCV)

    def test_model_has_predict_methods(self):
        """Test that model has required methods."""
        model = load_model()
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
        assert callable(model.predict)
        assert callable(model.predict_proba)
