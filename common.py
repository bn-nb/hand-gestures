import cv2 as cv
import numpy as np

from mediapipe import Image, ImageFormat, solutions
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.framework.formats import landmark_pb2

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path='assets/hand_landmarker.task',
    ),
    num_hands=1,
    running_mode=vision.RunningMode.IMAGE,
)
detector = vision.HandLandmarker.create_from_options(options)

H = 720
W = 1280

_BLACK_ = (0,0,0)
_WHITE_ = (255,255,255)
_VIOLA_ = (150,150,255)
_GREEN_ = (0,255,0)
__RED__ = (255,0,0)


def draw_bbox(image, color_tuple, x0, y0, x1, y1):
    cv.rectangle(
        image, 
        pt1=(x0-W//64,y0-H//36),
        pt2=(x1+W//64,y1+H//36),
        color=color_tuple, 
        thickness=2
    )


def put_title(image, title, color_tuple=_WHITE_):
    
    cv.rectangle(
        image, 
        pt1=(W//4,0),
        pt2=(3*W//4,H//15),
        color=(0,0,0), 
        thickness=-1
    )

    cv.putText(
        image, 
        text=title, 
        color=color_tuple, 
        org=(W//3-20, H//18-10), 
        thickness=2,
        fontFace=cv.FONT_HERSHEY_DUPLEX,
        fontScale=1, 
        lineType=cv.LINE_AA
    )


def put_labels(image, labels, x0, y0, x1, y1):

    label1 = 'HEURISTIC-'+labels[0]
    label2 = 'SKITMODEL-'+labels[1]

    box_start = y0-H//10
    box_end   = y0-H//36
    lab1_posn = (x0-10,y0-H//10-5)
    lab2_posn = (x0-10,y0-H//10+20)
    
    cv.rectangle(
        image, 
        pt1=(x0-W//64,box_start),
        pt2=(x1+W//64,box_end),
        color=_BLACK_, 
        thickness=-1
    )

    text_args = dict(
        img=image,
        thickness=1,
        fontFace=cv.FONT_HERSHEY_DUPLEX,
        fontScale=0.5, 
        lineType=cv.LINE_AA
        )
    lab2 = dict(text=label2, color=_VIOLA_, org=lab2_posn)
    lab1 = dict(text=label1, color=__RED__, org=lab1_posn)


    cv.putText(**text_args, **lab1)
    cv.putText(**text_args, **lab2)


def draw_hand_landmarks(image, landmarks):
    # These classes are bundled in google protocol buffers

    pb2_args = lambda landmark: dict(x=landmark.x, y=landmark.y, z=landmark.z)
    pb2_mark = landmark_pb2.NormalizedLandmark
    pb2_list = [ pb2_mark(**pb2_args(li)) for li in landmarks ]

    hand_proto = landmark_pb2.NormalizedLandmarkList()
    hand_proto.landmark.extend(pb2_list)

    solutions.drawing_utils.draw_landmarks(
        image,
        hand_proto,
        solutions.hands.HAND_CONNECTIONS,
        solutions.drawing_styles.get_default_hand_landmarks_style(),
        solutions.drawing_styles.get_default_hand_connections_style(),
    )


def attest_image(image, landmarks, *labels):

    _BLACK_ = (0,0,0)

    X = [int(landmark.x * W) for landmark in landmarks]
    Y = [int(landmark.y * H) for landmark in landmarks]

    coords = min(X), min(Y), max(X), max(Y)

    draw_hand_landmarks(image, landmarks)

    draw_bbox(image, _BLACK_, *coords)

    put_labels(image, labels, *coords)       


def detect(image):
    return detector.detect(Image(image_format=ImageFormat.SRGB, data=image))
