# 获得金字塔采样图像
import cv2

srcImage = cv2.imread('bottle1.jpg')
srcImage1 = srcImage
print(srcImage.shape[0],srcImage.shape[1])
ratio = 1
for i in range(3):
    srcImage = cv2.pyrDown(srcImage)
    ImageName = "py_"+str(i+1)+".jpg"
    cv2.imwrite(ImageName,srcImage)

cv2.imwrite("test.jpg",srcImage1)