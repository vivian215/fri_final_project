import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'C:/Users/vivian/Documents/fri_final_project/efficientdet_lite0.tflite'

BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path='C:/Users/vivian/Documents/fri_final_project/efficientdet_lite0.tflite'),
    max_results=5,
    running_mode=VisionRunningMode.IMAGE)

with ObjectDetector.create_from_options(options) as detector:
    # The detector is initialized. Use it here.
    mp_image = mp.Image.create_from_file('C:/Users/vivian/Documents/fri_final_project/catdog.jpg')

    # Perform object detection on the provided single image.
    detection_result = detector.detect(mp_image)

    print(detection_result)

def getClosestObject(fingerX, fingerY, fingerAngle):
    valid_things = []
    for thing in detection_result.detections:
        pass
        # if 
        # print(i)
        # print()

getClosestObject()