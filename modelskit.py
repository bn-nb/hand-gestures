import joblib
import numpy as np

model = joblib.load("assets/extratrees.joblib")
CLASSES_CAT_DECODE = ('PALM', 'FIST', 'SWAG', 'THREE', 'FOUR', 'PARTY', 'GUN', 'BLESS', 'L', 'DUMMY')

def get_label(XY_list):

    flat = np.zeros((42,))
    flat[:21] = XY_list[:, 0]
    flat[21:] = XY_list[:, 1]

    pred = model.predict([flat])
    label = CLASSES_CAT_DECODE[int(pred[0])]
    return label
