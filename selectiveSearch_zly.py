# -*- coding: utf-8 -*-
from __future__ import division

#zly_add
from skimage import io
from PIL import Image
import numpy as np
import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy


# "Selective Search for Object Recognition" by J.R.R. Uijlings et al.
#
#  - Modified version with LBP extractor for texture vectorization

"""


"""
def getInitRegions(img, scale, sigma, min_size):
    """
        实现算法：
            Input: (colour) image
            Output: Set of object location hypotheses L
            Obtain initial regions R = {r1, · · · , rn} using [13]
        获得原始的分割小区域R

        利用skimage.segmentation.felzenszwalb()函数
        Parameters:
            image : (width, height, 3) or (width, height) ndarray
        Input image.
            scale : float 越大，所得到的区域块儿更大，数量更少
                Free parameter. Higher means larger clusters.
            sigma : float 负责在分割前对图像进行高斯滤波去噪
                Width of Gaussian kernel used in preprocessing.
            min_size : int 区域最少应包含的像素点，如果某区域R分割后像素点个数小于min_size，则选择与R差异最小的区域进行合并
                Minimum component size. Enforced using postprocessing.
            multichannel : bool, optional (default: True)
                Whether the last axis of the image is to be interpreted as multiple channels. A value of False, for a 3D image, is not currently supported.
        Returns:
            segment_mask : (width, height) ndarray
                Integer mask indicating segment labels.
    """
    im_mask = skimage.segmentation.felzenszwalb(skimage.util.img_as_float(img), scale=scale, sigma=sigma,min_size=min_size)

    # img本身是三维的，(width,height,channel),此处将img增加一维，即给每一个像素一个初始的类别label
    # 将felsenszwalb函数得到的im_mask加在新增的img的第四维
    print(type(img))
    img = numpy.append(img, numpy.zeros(img.shape[:2])[:, :, numpy.newaxis], axis=2)
    img[:, :, 3] = im_mask

    return img


def calColorSim(r1, r2):
    """
        计算区域ri和r2的颜色相似度
        按照论文中的方法：
        对于3通道图像，取n=75的直方图，取r1与r2对应的n个颜色直方图["hist_c"]中n个较小的值求和，即得颜色相似度
    """
    return sum([min(a, b) for a, b in zip(r1["hist_c"], r2["hist_c"])])


def calTexttureSim(r1, r2):
    """
        计算区域r1与r2的纹理相似度
        和求颜色相似度方法相同，取纹理直方图[hist_t]找最小值求和
    """
    return sum([min(a, b) for a, b in zip(r1["hist_t"], r2["hist_t"])])


def calSizeSim(r1, r2, imsize):
    """
        计算区域r1和r2的尺寸相似度
        论文中公式：Ssize(ri,rj) = 1 - (size(ri)+size(rj))/size(im)
        imsize：size(im) denotes the size of the image in pixels.
    """
    return 1.0 - (r1["size"] + r2["size"]) / imsize


def calFillSim(r1, r2, imsize):
    """
        计算填充相似度，可以理解为是否将两区域进行合并的一个指标
        sfill(ri, r j) measures how well region ri and r j fit into each other.
        比如，如果r1包含了r2，则理应将二者合并为一个区域，避免出现hole(空洞)
        如果r1和r2相隔甚远，则不应将二者合并。
        bbsize：we define BBij to be the tight bounding box around ri and rj
            即能将两个区域均包含在内的紧致的boundingbox的size.
    """
    bbsize = (
        (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
        * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    )
    #论文中的公式：fill(ri,rj) = 1 - (size(BBij)-size(ri)-size(rj))/size(im)
    return 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize


def calTotalSim(r1, r2, imsize):
    """
    计算总的相似度：s(ri, r j) = a1Scolour(ri, r j)+a2Stexture(ri, r j)+a3Ssize(ri, r j)+a4Sfill(ri, r j),
    此处取权重均为1
    :param r1:区域r1
    :param r2:区域r2
    :param imsize:img大小
    :return:int
    """
    return (calColorSim(r1, r2) + calTexttureSim(r1, r2)
            + calSizeSim(r1, r2, imsize) + calFillSim(r1, r2, imsize))


def calColorHist(img):
    """
        对于每个区域，计算颜色直方图
        取bins=25，将一个颜色空间划分为25个小区间（bins越多，直方图对颜色分辨率越强）
    """

    BINS = 25
    hist = numpy.array([])

    for colour_channel in (0, 1, 2):

        # 提取一个颜色通道
        c = img[:, colour_channel]

        # 对每一个颜色通道计算颜色直方图
        hist = numpy.concatenate(
            [hist] + [numpy.histogram(c, BINS, (0.0, 255.0))[0]])

    # L1正则化
    hist = hist / len(img)

    return hist


def calTexttureGradient(img):
    """
        对每一幅图像计算纹理梯度
        原始的SelectiveSearch算法是在像素的八个方向上应用高斯生成。本篇论文使用了LBP（局部二值模式）进行替代
        output：[height(*)][width(*)]

        skimage.feature.local_binary_pattern()
        Parameters:
            image : (N, M) array
                Graylevel image.
            P : int
                Number of circularly symmetric neighbour set points (quantization of the angular space).
            R : float
                Radius of circle (spatial resolution of the operator).
            Returns:
                output : (N, M) array
                    LBP image.
    """
    ret = numpy.zeros((img.shape[0], img.shape[1], img.shape[2]))

    for colour_channel in (0, 1, 2):
        ret[:, :, colour_channel] = skimage.feature.local_binary_pattern(
            img[:, :, colour_channel], 8, 1.0)

    return ret


def calTexttureHist(img):
    """
        计算每个区域的纹理直方图
        calculate the histogram of gradient for each colours
        the size of output histogram：BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    """
    BINS = 10

    hist = numpy.array([])

    for colour_channel in (0, 1, 2):

        # 提取每个通道
        fd = img[:, colour_channel]

        # 计算各个方向的直方图，将它们合并，并加入结果集
        hist = numpy.concatenate(
            [hist] + [numpy.histogram(fd, BINS, (0.0, 1.0))[0]])

    # L1 归一化
    hist = hist / len(img)

    return hist


def extractRegions(img):
    """
    提取区域
    :param img:
    :return:
    """
    R = {}

    #RGB转HSV
    hsv = skimage.color.rgb2hsv(img[:, :, :3])

    # 计算像素点位置
    for y, i in enumerate(img):

        for x, (r, g, b, l) in enumerate(i):

            # initialize a new region
            if l not in R:
                R[l] = {
                    "min_x": 0xffff, "min_y": 0xffff,
                    "max_x": 0, "max_y": 0, "labels": [l]}

            # bounding box
            if R[l]["min_x"] > x:
                R[l]["min_x"] = x
            if R[l]["min_y"] > y:
                R[l]["min_y"] = y
            if R[l]["max_x"] < x:
                R[l]["max_x"] = x
            if R[l]["max_y"] < y:
                R[l]["max_y"] = y

    # pass 2: 计算纹理梯度
    tex_grad = calTexttureGradient(img)

    # pass 3: 计算每个区域的纹理梯度
    for k, v in list(R.items()):

        # 计算颜色直方图
        masked_pixels = hsv[:, :, :][img[:, :, 3] == k]
        R[k]["size"] = len(masked_pixels / 4)
        R[k]["hist_c"] = calColorHist(masked_pixels)

        # 计算纹理直方图
        R[k]["hist_t"] = calTexttureHist(tex_grad[:, :][img[:, :, 3] == k])

    return R


def extractNeighbours(regions):

    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    R = list(regions.items())
    neighbours = []
    for cur, a in enumerate(R[:-1]):
        for b in R[cur + 1:]:
            if intersect(a[1], b[1]):
                neighbours.append((a, b))

    return neighbours


def mergeRegions(r1, r2):
    """
    合并区域
    :param r1:region r1
    :param r2: region r2
    :return: new region
    """
    new_size = r1["size"] + r2["size"]
    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size,
        "hist_c": (
            r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size,
        "hist_t": (
            r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size,
        "labels": r1["labels"] + r2["labels"]
    }
    return rt


def selectiveSearch(img, scale=1, sigma=0.8, min_size=50):
    """
    Selective Search

    Parameters
    ----------
        img : ndarray
            Input image
        scale : int
            Free parameter. Higher means larger clusters in felzenszwalb segmentation.
        sigma : float
            Width of Gaussian kernel for felzenszwalb segmentation.
        min_size : int
            Minimum component size for felzenszwalb segmentation.
    Returns
    -------
        img : ndarray
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions : array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
            ]
    """

    # 确保img为3通道
    assert img.shape[2] == 3, "3ch image is expected"

    # 得到分割的原始最小区域
    # 区域label存储在img的第四维中 [r,g,b,(region)]
    img = getInitRegions(img, scale, sigma, min_size)

    if img is None:
        return None, {}

    #imsize为img大小
    imsize = img.shape[0] * img.shape[1]
    #获得初始分割区域R
    R = extractRegions(img)

    # 获得neighbor区域
    neighbours = extractNeighbours(R)

    # 计算各个原始区域的相似度
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = calTotalSim(ar, br, imsize)

    # 搜索
    while S != {}:

        # 得到相似度最大的两个区域
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]

        # 合并相应区域
        t = max(R.keys()) + 1.0
        R[t] = mergeRegions(R[i], R[j])

        #移除被合并区域的信息
        key_to_delete = []
        for k, v in list(S.items()):
            if (i in k) or (j in k):
                key_to_delete.append(k)

        for k in key_to_delete:
            del S[k]

        #计算新的区域集R 的相似度信息
        for k in [a for a in key_to_delete if a != (i, j)]:
            n = k[1] if k[0] in (i, j) else k[0]
            S[(t, n)] = calTotalSim(R[t], R[n], imsize)

    regions = []
    for k, r in list(R.items()):
        regions.append({
            'rect': (
                r['min_x'], r['min_y'],
                r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
            'size': r['size'],
            'labels': r['labels']
        })

    return img, regions

def main():
    # loading astronaut image
    # img = skimage.data.astronaut()
    str= input("请输入要处理的图像(绝对路径)：")
    img = Image.open(str)

    img.save('result.jpg', quality=100)
    img = np.asarray(img)
    # io.imshow(img)
    # print(type(img))  # 显示类型
    # print(img.shape)  # 显示尺寸
    # print(img.shape[0])  # 图片高度
    # print(img.shape[1])  # 图片宽度
    # print(img.shape[2])  # 图片通道数
    # print(img.size)  # 显示总像素个数
    # print(img.max())  # 最大像素值
    # print(img.min())  # 最小像素值
    # print(img.mean())  # 像素平均值
    # print(img[0][0])  # 图像的像素值

    img_lbl, regions = selectiveSearch(
        img, scale=200, sigma=0.9, min_size=10)

    candidates = set()
    for r in regions:
        # 去除相似的矩形框
        if r['rect'] in candidates:
            continue
        # 去除像素值小于2000的矩形框
        if r['size'] < 2000:
            continue
        # 去除形状过“细”的矩形框
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])

    # 绘制矩形框
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in candidates:
        print(x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.axis('off')#关闭坐标轴刻度
    plt.show()

if __name__ == "__main__":
    main()
