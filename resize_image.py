import cv2
import numpy as np

INPUT_SIZE = 72

def resize_image(srcImage):
    height, width, channel = srcImage.shape

    # 计算图像像素均值
    gray_image = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
    mat_mean, mat_stddev = cv2.meanStdDev(gray_image)
    mean = mat_mean[0][0]
    # 建立新图像
    res = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.uint8)
    # 使用均值进行填充
    res.fill(int(mean))

    if height<width:    # 高度小于宽度，对齐后以高度的中线为基线进行对齐
        ratio = width / INPUT_SIZE
        res_height = int(height / ratio)
        res_temp = cv2.resize(srcImage,(INPUT_SIZE, res_height))
        # 实现 copyto 的功能
        np_res_temp = np.array(res_temp)
        start_height = 35 - res_height//2
        end_height = start_height+res_height
        res[start_height:end_height, 0:INPUT_SIZE] = np_res_temp

    elif height > width: # 高度大于宽度，对齐后以宽度的中线为基线进行对齐
        ratio = height / INPUT_SIZE
        res_width = int(width / ratio)
        res_temp = cv2.resize(srcImage, (res_width, INPUT_SIZE))
        # 实现 copyto 的功能
        np_res_temp = np.array(res_temp)
        start_width = 35 - res_width // 2
        end_width = start_width + res_width
        res[0:72, start_width:end_width] = np_res_temp

    else: # 高度等于宽度
        res = cv2.resize(srcImage,(INPUT_SIZE,INPUT_SIZE))
    return res

# srcImage1 = cv2.imread('test1.jpg')
# srcImage1 = resize_image(srcImage1)
# cv2.imshow('pp',srcImage1)
# cv2.waitKey()