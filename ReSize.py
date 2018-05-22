import os
import cv2
from resize_image import *

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, 'per')

objectPath = []
objectDiffPath = []
objectDiffPath_Path = []

for f in os.listdir(DATA_PATH):
    objectPath.append(DATA_PATH + "\\" + f)

numOfClass = len(objectPath)
for i in range(numOfClass):
    object_path = objectPath[i]
    for ff in os.listdir(object_path):
        objectDiffPath.append(object_path + "\\" + ff)

for j in range(len(objectDiffPath)):
    objectDiff_path = objectDiffPath[j]
    for fff in os.listdir(objectDiff_path):
        objectDiffPath_Path.append(objectDiff_path + "\\" + fff)

test = []
for i in range(len(objectDiffPath_Path)):
    img = cv2.imread(objectDiffPath_Path[i])
    img = resize_image(img)
    #img2 = cv2.resize(img, (72, 72), interpolation=cv2.INTER_AREA)
    cv2.imwrite(objectDiffPath_Path[i], img)