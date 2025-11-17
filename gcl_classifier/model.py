from pathlib import Path
import skops.io as sio


def load_model():
    model_path = Path(__file__).parent / "weights" / "gcl_classifier.skops"
    obj = sio.load(model_path, trusted=[
        'sklearn.calibration._CalibratedClassifier',
        'sklearn.calibration._SigmoidCalibration'
    ])

    return obj['model']

model = load_model()
