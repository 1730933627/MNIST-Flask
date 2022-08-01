import numpy as np
import base64
import cv2
import math
from scipy import ndimage


def preprocess(img):
    """
    : param img : 输入黑白图像
    : returns img : 处理过的图像
    """
    return cv2.resize(img, (28, 28))
    # 裁剪出完全白色的边缘
    while int(np.mean(img[0])) == 255:
        img = img[1:]
    while np.mean(img[:, 0]) == 255:
        img = np.delete(img, 0, 1)
    while np.mean(img[-1]) == 255:
        img = img[:-1]
    while np.mean(img[:, -1]) == 255:
        img = np.delete(img, -1, 1)
    # 调整到合适的形状
    rows, cols = img.shape
    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        img = cv2.resize(img, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        img = cv2.resize(img, (cols, rows))

    # 添加预定义的填充
    colsPadding = (int(math.ceil((28 - cols) / 2.0)),
                   int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)),
                   int(math.floor((28 - rows) / 2.0)))
    img = np.lib.pad(img, (rowsPadding, colsPadding),
                     'constant', constant_values=255)
    # 移动图像,使数字的质心,很好地居中
    shiftx, shifty = getBestShift(img)
    shifted = shift(img, shiftx, shifty)
    img = shifted
    cv2.imwrite('static/user_drawn/temp.png', img)
    return img


def getBestShift(img):
    """
    : param img : 一个数字的黑白图像
    : returns img : 最佳移位 (x, y)
    """
    # 计算质量中心
    cy, cx = ndimage.measurements.center_of_mass(img)
    # 计算质心与实际的差异,图像中心以获得转变
    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)
    return shiftx, shifty


def shift(img, sx, sy):
    """
    将图像移动一些偏移量
    : param img : 黑白图像
    : param sx  : x 方向位移
    : param sy  : y 方向位移
    : returns   : 位移后的图像
    """
    # 生成仅表示平移（旋转）的变形矩阵,部分是单位矩阵
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    # 应用变形矩阵
    rows, cols = img.shape
    shifted = cv2.warpAffine(img, M, (cols, rows),
                             borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return shifted


def data_uri_to_cv2_img(uri):
    """
    将数据 URL 转换为 OpenCV 图像
    : param uri : 表示 BW 图像的数据 URI
    : returns   : OpenCV图像
    """
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    return img


def value_invert(array):
    """
    接受一个数组，假设包含 0 到 1 之间的值，然后反转那些具有转换 x -> 1 - x 的值。
    """
    # 将数组展平以进行循环
    flatarray = array.flatten()
    # 将变换应用于展平数组
    for i in range(flatarray.size):
        flatarray[i] = 1 - flatarray[i]
    # 返回具有原始形状的转换后的数组
    return flatarray.reshape(array.shape)
