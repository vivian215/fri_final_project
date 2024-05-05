import yolo_obj_detect as yolo
import gesture_recognizer_still as gesture
import sys
import numpy as np
import cv2
from shapely.geometry import LineString, Point, Polygon, box
import matplotlib.pyplot as plt
import math

def recognize_object(imagePath):
    #gets all objects inside image
    image = cv2.imread(imagePath)
    results = yolo.getAllObjects(imagePath)
    boxes_og = results[0].boxes
    classes_og = results[0].boxes.cls
    names = []
    boxes = []
    classes = []
    for i in range(len(classes_og)):
        if classes_og[i] != 0:
            classes.append(classes_og[i])
            boxes.append(boxes_og.xyxy[i])
            names.append(yolo.model.names[int(classes_og[i])])
    # for c in classes:
        # if c != 0: #get rid of persons
        # names.append(yolo.model.names[int(c)])

    #analyzes the finger inside the image
    gesture.createFinger(imagePath)

    #gets intersection between a ray and rectangle bounding box
    def getIntersection(ray_start, ray_end, rect):
        # fig, ax = plt.subplots()
        center_y = image.shape[0]/2
        center_x = image.shape[1]/2

        polygon = Polygon(box(rect[0]-center_x, rect[1]-center_y, rect[2]-center_x, rect[3]-center_y))
        
        line = LineString([(ray_start[0], ray_start[1]), (ray_end[0], ray_end[1])])
        
        intersection = line.intersection(polygon)
        # print(intersection)

        # ax.plot(*line.xy)
        # ax.plot(*polygon.exterior.xy)
        # # ax.plot(intersection.x, intersection.y, 'pt')

        # print(line)
        # print(polygon)

        # ax.set_xlim(-1*center_x, center_x)
        # ax.set_ylim(-1*center_y, center_y)
        # ax.set_aspect('equal', adjustable='box')
        
        # Show the plot
        # plt.show()
        # return isinstance(intersection, Point)
        return intersection

    #determines if p1 or p2 is closer to p
    def getCloserPoint(p, p1, p2):
        p1 = np.array(p1)
        p2 = np.array(p2)
        d1 = math.sqrt((p1[0] - p[0]) ** 2 + (p1[1] - p[1]) ** 2)
        d2 = math.sqrt((p2[0] - p[0]) ** 2 + (p2[1] - p[1]) ** 2)
        return p1 if d1 < d2 else p2
    def getCloserPointIdx(p, p1, p2):
        p1 = np.array(p1)
        p2 = np.array(p2)
        d1 = math.sqrt((p1[0] - p[0]) ** 2 + (p1[1] - p[1]) ** 2)
        d2 = math.sqrt((p2[0] - p[0]) ** 2 + (p2[1] - p[1]) ** 2)
        return 1 if d1 < d2 else 2

    #returns object closest to finger point within its direction
    def findNearestObj():
        if (not gesture.isStraight): 
            return None
        
        #found a finger pointing -> get ray that extends in direction of finger
        point1 = gesture.getP5()
        point2 = gesture.getP8()
        direction = np.array([point2[0] - point1[0], point2[1] - point1[1]])
        magnitude = np.linalg.norm(direction)
        unit_vector = direction / magnitude
        ray_start = np.array([point2[0], point2[1]])
        SCALE = 5000
        ray_end = np.array([ray_start[0] + unit_vector[0] * SCALE, ray_start[1] + unit_vector[1] * SCALE])

        #gets closest object out of all objects the point intersects
        closestPoint = (0,0)
        closestIdx = -1
        for i in range(len(boxes)):
            intersection = getIntersection(ray_start, ray_end, boxes[i])
            if i == 0:
                if isinstance(intersection, Point):
                    closestPoint = intersection
                    closestIdx = 0
                elif isinstance(intersection, LineString):
                    closestPoint = getCloserPoint(point2, intersection.coords[0], intersection.coords[1])
                    closestIdx = 0
            else:
                if isinstance(intersection, Point):
                    closestPoint = getCloserPoint(point2, closestPoint, intersection)
                    closestIdx = closestIdx if getCloserPointIdx(point2, closestPoint, intersection) == 1 else i
                elif isinstance(intersection, LineString):
                    closerPointInLine = getCloserPoint(point2, intersection.coords[0], intersection.coords[1])
                    closestPoint = getCloserPoint(point2, closestPoint, closerPointInLine)
                    closestIdx = closestIdx if getCloserPointIdx(point2, closestPoint, closerPointInLine) == 1 else i

        if closestIdx < 0:
            return "none"
        return names[closestIdx]

    return findNearestObj




