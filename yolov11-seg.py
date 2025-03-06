import cv2 
from ultralytics import YOLO
import random
import numpy as np

model = YOLO('yolo11x-seg.pt')
img = cv2.imread('data/2.jpg')

yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
conf = 0.2
results = model.predict(img, conf=conf)
colors = [random.choices(range(256), k=3) for _ in classes_ids]
for result in results:
    for mask, box in zip(result.masks.xy, result.boxes):
        points = np.int32([mask])
        color_number = classes_ids.index(int(box.cls[0]))
        cv2.fillPoly(img, points, colors[color_number])

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.imwrite('Saved', img)