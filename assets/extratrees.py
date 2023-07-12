# Imports and setup

import numpy
import joblib

# GridSearchCV Results: ExtraTreesClassifier gave best results for this data.

from matplotlib       import pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics  import ConfusionMatrixDisplay
from sklearn.metrics  import confusion_matrix


CLASSES_CAT_DECODE = ('PALM', 'FIST', 'SWAG', 'THREE', 'FOUR', 'PARTY', 'GUN', 'BLESS', 'L', 'DUMMY')
df = numpy.load('./train.npz')['train']

X, y = df[:, :42], df[:, 42]
print(X.shape, y.shape)


# Model training

trees = ExtraTreesClassifier().fit(X,y)
joblib.dump(trees, "extratrees.joblib", compress=3)

fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay(
    confusion_matrix(y, trees.predict(X)),
    display_labels=CLASSES_CAT_DECODE,
).plot(ax=ax)

plt.show()
