from ultralytics import YOLO

#pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Define path to the image file
def getAllObjects(imagePath):
    return model(imagePath)