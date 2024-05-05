import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2

#set up model and model parameters
model_path = 'C:/Users/vivian/Documents/fri_final_project/hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

#get angle between two vectors
def getAngleBetween(v1, v2):
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cos_theta)  # in radians
    return np.degrees(angle)  # convert to degrees

#determine if two angles connect in a straight line
def isStraight(angle1, angle2):
    STRAIGHTNESS_THRESHOLD = 12
    return angle1 < STRAIGHTNESS_THRESHOLD and angle2 < STRAIGHTNESS_THRESHOLD

#gets the direction of a finger point
def getFingerAngle(v1, v2, v3):
    x_ref = np.array([1, 0, 0]) #left/right
    y_ref = np.array([0, 1, 0]) #forward/backward
    x_avg_angle = (getAngleBetween(v1, x_ref) + getAngleBetween(v2, x_ref) + getAngleBetween(v3, x_ref)) / 3
    y_avg_angle = (getAngleBetween(v1, y_ref) + getAngleBetween(v2, y_ref) + getAngleBetween(v3, y_ref)) / 3

    #direction is 1 if left/right, 0 if forward/backward
    if x_avg_angle > 45 and x_avg_angle < 135 or x_avg_angle > 225 and x_avg_angle < 315:
        direction = 0
        actual_angle = y_avg_angle
    else:
        direction = 1
        actual_angle = x_avg_angle

    #determine final direction
    THRESHOLD = 10
    if (actual_angle > 90 + THRESHOLD):
        final_direction = "left" if direction == 1 else "forward"
    elif (actual_angle < 90 - THRESHOLD):
        final_direction = "right" if direction == 1 else "invalid"
    else:
        final_direction = "invalid"
    return final_direction

p5 = 0
p8 = 0

#gets positions of a hand given an image
def createFinger(imagePath):
    global p5, p8
    #create hand landmarker
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='C:/Users/vivian/Documents/fri_final_project/hand_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE)
    with HandLandmarker.create_from_options(options) as landmarker:

        #load and detect hand in input image
        mp_image = mp.Image.create_from_file(imagePath) 
        hand_landmarker_result = landmarker.detect(mp_image)

        result5 = hand_landmarker_result.hand_landmarks[0][5]
        result6 = hand_landmarker_result.hand_landmarks[0][6]
        result7 = hand_landmarker_result.hand_landmarks[0][7]
        result8 = hand_landmarker_result.hand_landmarks[0][8]

        #get all points on index finger
        p5 = np.array([result5.x, result5.y, result5.z])
        p6 = np.array([result6.x, result6.y, result6.z])
        p7 = np.array([result7.x, result7.y, result7.z])
        p8 = np.array([result8.x, result8.y, result8.z])

        #get vectors between each point
        v1 = p6 - p5
        v2 = p7 - p6
        v3 = p8 - p7

        return [v1, v2, v3]

#gets direction of finger given image
def getFingerDirection(imagePath):
    finger_vecs = createFinger(imagePath)
    return getFingerAngle(finger_vecs[0], finger_vecs[1], finger_vecs[2])

def getP5():
    return p5

def getP8():
    return p8