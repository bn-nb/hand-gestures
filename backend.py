import cv2   as cv
import numpy as np
import copy
import pickle

from sklearn.preprocessing import MinMaxScaler

import common
import heuristic as heur
import modelskit as skml

MMS = MinMaxScaler()

def normalize(landmarks):
    cpy = copy.deepcopy(landmarks)
    tmp = np.array([[l.x, l.y] for l in cpy])
    # Rotation Matrix
    rad = np.arctan2(tmp[9][0] - tmp[0][0], tmp[9][1] - tmp[0][1]) + np.pi
    mat = np.array([[np.cos(rad), -np.sin(rad)],[np.sin(rad), np.cos(rad)]])
    # Rotate to Center + Scale
    tmp = (mat @ (tmp-0.5).T + 0.5).T
    tmp = MMS.fit_transform(tmp) * 0.8 + 0.1

    for i in range(21):
        cpy[i].x, cpy[i].y = tmp[i]

    return cpy, tmp


def process(BGR_image):
    image = cv.flip( cv.cvtColor(BGR_image, cv.COLOR_BGR2RGB), 1 )
    result = common.detect(image)
    common.put_title(image, "HAND GESTURE RECOGNITION")

    for i in range(len(result.hand_landmarks)):

        land_i = result.hand_landmarks[i]
        hand_i = result.handedness[i][0].category_name
        label1 = hand_i + '-' + heur.get_label(normalize(land_i)[1])
        label2 = hand_i + '-' + skml.get_label(normalize(land_i)[1])
        common.attest_image(image, land_i, label1, label2)

    return cv.cvtColor(image, cv.COLOR_RGB2BGR)


def record_data():
    # 9 classes + 1 Dummy
    SAMPLES_PER_CLASS = 250
    NUMBER_OF_CLASSES = 10
    CLASSES_CAT_ENCODE = dict(PALM=0, FIST=1, SWAG=2, THREE=3, FOUR=4, PARTY=5, GUN=6, BLESS=7, L=8, DUMMY=9)
    CLASSES_DATA_COUNT = dict(PALM=0, FIST=0, SWAG=0, THREE=0, FOUR=0, PARTY=0, GUN=0, BLESS=0, L=0, DUMMY=0)
    dataset = np.zeros((SAMPLES_PER_CLASS * NUMBER_OF_CLASSES, 21+21+1))
    index = 0

    H = 360
    W = 640

    vid_i = cv.VideoCapture(0)
    vid_i.set(3, W)
    vid_i.set(4, H)
    temp, data = None, None
    
    while (read_image := vid_i.read())[0]:
        img_0 = cv.flip( cv.cvtColor(read_image[1], cv.COLOR_BGR2RGB), 1 )
        img_1 = np.zeros((H, H, 3))
        marks = common.detect(img_0)

        for mark in marks.hand_landmarks:
            new_mark, xy_list = normalize(mark)
            common.draw_hand_landmarks(img_1, new_mark)
            label = heur.get_label(xy_list)
            index = int(sum(CLASSES_DATA_COUNT.values()))

            if (CLASSES_DATA_COUNT[label] < SAMPLES_PER_CLASS):
                CLASSES_DATA_COUNT[label] += 1
                dataset[index, 0:21]  = xy_list[:, 0]
                dataset[index, 21:42] = xy_list[:, 1]
                dataset[index, 42]    = CLASSES_CAT_ENCODE[label]
            else:
                print(f"Label: {label} is full! Change to record new data")

        cv.namedWindow("TEST")
        cv.moveWindow("TEST", 1000, 80)
        cv.imshow("TEST", img_1)
        key = chr(cv.waitKey(1) % 256)

        if key in 'qQ':
            break
        elif key in 'pP':
            print(data)
        elif key in 'sS':
            print(heur.FingerState(data).get_state())

        if ((index+1) == dataset.shape[0]):
            print("Data Recording completed!")
            break
        elif not ((index+1) % 100):
            print(CLASSES_DATA_COUNT)

    cv.destroyAllWindows()

    with open('assets/training_0.npy', 'wb') as file:
        np.save(file, dataset)


if __name__ == "__main__":
    record_data()
