# 获得金字塔采样图像
import cv2

srcImage = cv2.imread('bottle1.jpg')

print(srcImage.shape[0],srcImage.shape[1])
ratio = 1
for i in range(3):
    ratio *= 2
    srcImage = cv2.pyrDown(srcImage,(srcImage.shape[1]/ratio,srcImage.shape[0]/ratio))
    ImageName = "py_"+str(i+1)+".jpg"
    cv2.imwrite(ImageName,srcImage)
