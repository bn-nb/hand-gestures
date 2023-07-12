## SETUP - TL;DR

* Install the necessary python packages, using
>  ``> pip install -r ./docs/requirements.txt``

* On Linux: Install ``./docs/packages.txt`` **only if needed**, using
>``> sudo <package manager> install $(cat ./docs/packages.txt)``

* To start the app, call from this directory:
>``> python frontend.py``

## BACKGROUND

* This project utilizes MediaPipe's handlandmarker to identify hand keypoints.
* It then uses the keypoints to classify the hand gestures (9 classes  + 1 dummy).
* Heuristic method uses rigid geometric rules to identify gestures.
* Skitmodel method uses a scikit model trained from heuristics data.
* Check ``docs/states.png`` for the supported gestures with encoding.
* The ``assets/`` folder stores:
   * the pre-trained mediapipe handlandmarker model from google.
   * our ExtraTreesClassifier model, with the training script.
   * the training data (10000 examples) in a ```.npz``` file.
* To collect more training data, run  ``backend.py``  as ``__main__``

## CHECKS

* App may take (5s to 20s) to start running, depending on the system.
* Try to use a static, contrasting background for your hands.
* This ensures that the mediapipe framework is able to detect hands easily.
