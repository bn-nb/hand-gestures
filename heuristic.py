import numpy as np
import cv2   as cv

OPEN   = 0
BENT   = 1
CLOSED = 2

BEND_THRESHOLD = 140


def eval_line(x0, y0, x1, y1):
    m = (y1-y0)/(x1-x0)
    c = y0 - m*x0
    def func(x, y):
        return m * x + c - y
    return func


def is_same_side(linept1, linept2, point1, point2):
    # Check if pt1 & pt2 lie on the same side of
    # the line formed by lpt1 and lpt2
    line = eval_line(*linept1, *linept2)
    z1 = line(*point1)
    z2 = line(*point2)
    return (z1/z2) > 0


def dist_btwn(pt1, pt2):
    d_x2 = (pt1[0] - pt2[0]) ** 2
    d_y2 = (pt1[1] - pt2[1]) ** 2
    d_r = (d_x2 + d_y2) ** 0.5
    return d_r


def angle_at(A, B, C):
    # Find angle between AB and BC, formed at B
    a = dist_btwn(B,C)
    b = dist_btwn(A,C)
    c = dist_btwn(A,B)

    rad = np.arccos( (a**2 + c**2 - b**2) / (2*a*c) )
    deg = rad / np.pi * 180
    return deg


class FingerState:
    def __init__(self, marks):
        self.marks = marks

    def thumb_state(self):
        angles = [
            angle_at(*self.marks[[0,1,2],]),
            angle_at(*self.marks[[1,2,3],]),
            angle_at(*self.marks[[2,3,4],]),
        ]

        if (is_same_side( *self.marks[[0,5,9,4]] )):
            return CLOSED
        elif (np.min(angles) > BEND_THRESHOLD):
            return OPEN
        else:
            return BENT

    def index_state(self):
        angles = [
            # angle_at(*self.marks[[0,5,6],]),
            angle_at(*self.marks[[5,6,7],]),
            angle_at(*self.marks[[6,7,8],]),
        ]

        if (is_same_side( *self.marks[[17,5,0,8]] )):
            return CLOSED
        elif (np.min(angles) > BEND_THRESHOLD):
            return OPEN
        else:
            return BENT

    def middle_state(self):
        angles = [
            # angle_at(*self.marks[[0,9,10],]),
            angle_at(*self.marks[[9,10,11],]),
            angle_at(*self.marks[[10,11,12],]),
        ]

        if (is_same_side( *self.marks[[17,5,0,12]] )):
            return CLOSED
        elif (np.min(angles) > BEND_THRESHOLD):
            return OPEN
        else:
            return BENT

    def ring_state(self):
        angles = [
            # angle_at(*self.marks[[0,13,14],]),
            angle_at(*self.marks[[13,14,15],]),
            angle_at(*self.marks[[14,15,16],]),
        ]

        if (is_same_side( *self.marks[[17,5,0,16]] )):
            return CLOSED
        elif (np.min(angles) > BEND_THRESHOLD):
            return OPEN
        else:
            return BENT

    def pinky_state(self):
        angles = [
            # angle_at(*self.marks[[0,17,18],]),
            angle_at(*self.marks[[17,18,19],]),
            angle_at(*self.marks[[18,19,20],]),
        ]

        if (is_same_side( *self.marks[[17,5,0,20]] )):
            return CLOSED
        elif (np.min(angles) > BEND_THRESHOLD):
            return OPEN
        else:
            return BENT

    def get_state(self):
        t=self.thumb_state()
        i=self.index_state()
        m=self.middle_state()
        r=self.ring_state()
        p=self.pinky_state()
        return f"{t}{i}{m}{r}{p}"


def get_label(XY_list):

    state = FingerState(XY_list).get_state()

    dicty = {
        "00000":"PALM",
        "22222":"FIST",
        "12222":"FIST",
        "12220":"SWAG",
        "02220":"SWAG",
        "20002":"THREE",
        "20000":"FOUR",
        "10220":"PARTY",
        "00220":"PARTY",
        "10022":"GUN",
        "00022":"GUN",
        "10000":"BLESS",
        "00222":"L",
    }

    label = dicty.get(state, "DUMMY")
    return label
