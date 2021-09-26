# encoding='UTF-8'

# 包括:
#     1. 改变亮度
#     2. 加噪声
#     3. 加随机点
#     4. 镜像(需要改变points)
#     5. 平移(需要改变points)
#     6. 旋转(需要改变points及图片尺寸)
#     7. 裁剪(需要改变points)

import time
import random
import cv2
import os
import numpy as np
from skimage.util import random_noise
import base64
import json
import re
from copy import deepcopy
import argparse
from math import cos, sin, pi, fabs, radians


# 图像均为cv2读取
class DataAugmentForObjectDetection():
    def __init__(self, change_light_rate=0.5, rotation_rate = 0.5, max_rotation_angle = 5, crop_rate = 0.5,
                 add_noise_rate=0.5, random_point=0.5, flip_rate=0.5, shift_rate=0.5, rand_point_percent=0.03,
                 is_addNoise=False, is_changeLight=True, is_random_point=False, is_shift_pic_bboxes=True,
                 is_filp_pic_bboxes=True, is_rotate_pic_bboxes = True, is_crop_pic_bboxes = True):
        # 配置各个操作的属性
        self.change_light_rate = change_light_rate
        self.add_noise_rate = add_noise_rate
        self.random_point = random_point
        self.flip_rate = flip_rate
        self.shift_rate = shift_rate
        self.rotation_rate = rotation_rate
        self.crop_rate = crop_rate
        self.max_rotation_angle = max_rotation_angle
        self.rand_point_percent = rand_point_percent

        # 是否使用某种增强方式
        self.is_addNoise = is_addNoise
        self.is_changeLight = is_changeLight
        self.is_random_point = is_random_point
        self.is_filp_pic_bboxes = is_filp_pic_bboxes
        self.is_shift_pic_bboxes = is_shift_pic_bboxes
        self.is_rotate_pic_bboxes= is_rotate_pic_bboxes
        self.is_crop_pic_bboxes = is_crop_pic_bboxes

    # 加噪声
    def _addNoise(self, img):
        return random_noise(img, seed=int(time.time())) * 255

    # 调整亮度
    def _changeLight(self, img):
        alpha = random.uniform(0.35, 1)
        blank = np.zeros(img.shape, img.dtype)
        return cv2.addWeighted(img, alpha, blank, 1 - alpha, 0)

    # 随机的改变点的值
    def _addRandPoint(self, img):
        percent = self.rand_point_percent
        num = int(percent * img.shape[0] * img.shape[1])
        for i in range(num):
            rand_x = random.randint(0, img.shape[0] - 1)
            rand_y = random.randint(0, img.shape[1] - 1)
            if random.randint(0, 1) == 0:
                img[rand_x, rand_y] = 0
            else:
                img[rand_x, rand_y] = 255
        return img

    # 平移
    def _shift_pic_bboxes(self, img, json_info):

        # ---------------------- 平移图像 ----------------------
        h, w, _ = img.shape
        x_min = w
        x_max = 0
        y_min = h
        y_max = 0

        shapes = json_info['shapes']
        for shape in shapes:
            points = np.array(shape['points'])
            x_min = min(x_min, points[:, 0].min())
            y_min = min(y_min, points[:, 1].min())
            x_max = max(x_max, points[:, 0].max())
            y_max = max(y_max, points[:, 0].max())

        d_to_left = x_min  # 包含所有目标框的最大左移动距离
        d_to_right = w - x_max  # 包含所有目标框的最大右移动距离
        d_to_top = y_min  # 包含所有目标框的最大上移动距离
        d_to_bottom = h - y_max  # 包含所有目标框的最大下移动距离

        x = random.uniform(-(d_to_left - 1) / 3, (d_to_right - 1) / 3)
        y = random.uniform(-(d_to_top - 1) / 3, (d_to_bottom - 1) / 3)

        M = np.float32([[1, 0, x], [0, 1, y]])  # x为向左或右移动的像素值,正为向右负为向左; y为向上或者向下移动的像素值,正为向下负为向上
        shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        # ---------------------- 平移boundingbox ----------------------
        for shape in shapes:
            for p in shape['points']:
                p[0] += x
                p[1] += y
        return shift_img, json_info

    # 镜像
    def _filp_pic_bboxes(self, img, json_info):

        # ---------------------- 翻转图像 ----------------------
        h, w, _ = img.shape

        sed = random.random()

        if 0 < sed < 0.33:  # 0.33的概率水平翻转，0.33的概率垂直翻转,0.33是对角反转
            flip_img = cv2.flip(img, 0)  # _flip_x
            inver = 0
        elif 0.33 < sed < 0.66:
            flip_img = cv2.flip(img, 1)  # _flip_y
            inver = 1
        else:
            flip_img = cv2.flip(img, -1)  # flip_x_y
            inver = -1

        # ---------------------- 调整boundingbox ----------------------
        shapes = json_info['shapes']
        for shape in shapes:
            for p in shape['points']:
                if inver == 0:
                    p[1] = h - p[1]
                elif inver == 1:
                    p[0] = w - p[0]
                elif inver == -1:
                    p[0] = w - p[0]
                    p[1] = h - p[1]

        return flip_img, json_info

    # 裁剪
    def _crop_pic_bboxes(self, img, json_info):

        # ---------------------- 裁剪图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w  # 裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h
        y_max = 0
        shapes = json_info['shapes']
        for shape in shapes:
            for p in shape['points']:
                x_min = min(x_min, p[0])
                y_min = min(y_min, p[1])
                x_max = max(x_max, p[0])
                y_max = max(y_max, p[1])

        d_to_left = x_min  # 包含所有目标框的最小框到左边的距离
        d_to_right = w - x_max  # 包含所有目标框的最小框到右边的距离
        d_to_top = y_min  # 包含所有目标框的最小框到顶端的距离
        d_to_bottom = h - y_max  # 包含所有目标框的最小框到底部的距离

        # 随机扩展这个最小框
        # crop_x_min = int(x_min - random.uniform(0, d_to_left))
        # crop_y_min = int(y_min - random.uniform(0, d_to_top))
        # crop_x_max = int(x_max + random.uniform(0, d_to_right))
        # crop_y_max = int(y_max + random.uniform(0, d_to_bottom))

        # 随机扩展这个最小框 , 防止别裁的太小
        crop_x_min = int(x_min - random.uniform(d_to_left // 2, d_to_left))
        crop_y_min = int(y_min - random.uniform(d_to_top // 2, d_to_top))
        crop_x_max = int(x_max + random.uniform(d_to_right // 2, d_to_right))
        crop_y_max = int(y_max + random.uniform(d_to_bottom // 2, d_to_bottom))

        # 确保不要越界
        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        # ---------------------- 裁剪boundingbox ----------------------
        # 裁剪后的boundingbox坐标计算
        shapes = json_info['shapes']
        for shape in shapes:
            for p in shape['points']:
                p[0] = p[0] - crop_x_min
                p[1] = p[1] - crop_y_min

        json_info['imageHeight'] = crop_img.shape[0]
        json_info['imageWidth'] = crop_img.shape[1]
        return crop_img, json_info

    # 旋转
    def _rotate_pic_bboxes(self, img, json_info, degree = 5):

        # ---------------------- 旋转图像 ----------------------
        height, width = img.shape[:2]
        heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
        widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
        matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
        matRotation[0, 2] += (widthNew - width) // 2
        matRotation[1, 2] += (heightNew - height) // 2
        # imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
        imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(0, 0, 0))

        # ---------------------- 调整boundingbox ----------------------
        json_dict = {}
        for key, value in json_info.items():
            if key == 'imageHeight':
                json_dict[key] = imgRotation.shape[0]
            elif key == 'imageWidth':
                json_dict[key] = imgRotation.shape[1]
            else:
                json_dict[key] = value
        for item in json_dict['shapes']:
            for key, value in item.items():
                if key == 'points':
                    for item2 in range(len(value)):
                        pt1 = np.dot(matRotation, np.array([[value[item2][0]], [value[item2][1]], [1]]))
                        value[item2][0], value[item2][1] = pt1[0][0], pt1[1][0]
        return imgRotation, json_dict

    # 图像增强方法
    def dataAugment(self, img, dic_info):

        change_num = 0  # 改变的次数
        while change_num < 1:  # 默认至少有一种数据增强生效

            if self.is_changeLight:
                if random.random() > self.change_light_rate:  # 改变亮度
                    change_num += 1
                    img = self._changeLight(img)

            if self.is_addNoise:
                if random.random() < self.add_noise_rate:  # 加噪声
                    change_num += 1
                    img = self._addNoise(img)
            if self.is_random_point:
                if random.random() > self.random_point:  # 加随机点
                    change_num += 1
                    img = self._addRandPoint(img)
            if self.is_shift_pic_bboxes:
                if random.random() < self.shift_rate:  # 平移
                    change_num += 1
                    img, dic_info = self._shift_pic_bboxes(img, dic_info)
            if self.is_filp_pic_bboxes:
                if random.random() > self.flip_rate:  # 翻转
                    change_num += 1
                    img, dic_info = self._filp_pic_bboxes(img, dic_info)
            if self.is_crop_pic_bboxes:
                if random.random() < self.crop_rate:  # 裁剪
                    change_num += 1
                    img, dic_info = self._crop_pic_bboxes(img, dic_info)
            if self.is_rotate_pic_bboxes:
                if random.random() > self.rotation_rate:  # 旋转
                    change_num += 1
                    angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
                    img, dic_info = self._rotate_pic_bboxes(img, dic_info, angle)

        return img, dic_info


# xml解析工具
class ToolHelper:
    # 从json文件中提取原始标定的信息
    def parse_json(self, path):
        with open(path)as f:
            json_data = json.load(f)
        return json_data

    # 对图片进行字符编码
    def img2str(self, img_name):
        with open(img_name, "rb")as f:
            base64_data = str(base64.b64encode(f.read()))
        match_pattern = re.compile(r'b\'(.*)\'')
        base64_data = match_pattern.match(base64_data).group(1)
        return base64_data

    # 保存图片结果
    def save_img(self, save_path, img):
        cv2.imwrite(save_path, img)

    # 保持json结果

    def save_json(self, file_name, save_folder, dic_info):
        with open(os.path.join(save_folder, file_name), 'w') as f:
            json.dump(dic_info, f, indent=2)


if __name__ == '__main__':

    need_aug_num = 20  # 每张图片需要增强的次数

    toolhelper = ToolHelper()  # 工具

    is_endwidth_dot = True  # 文件是否以.jpg或者png结尾

    dataAug = DataAugmentForObjectDetection()  # 数据增强工具类

    # 获取相关参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_img_json_path', type=str, default='data')
    parser.add_argument('--save_img_json_path', type=str, default='data2')
    args = parser.parse_args()
    source_img_json_path = args.source_img_json_path  # 图片和json文件原始位置
    save_img_json_path = args.save_img_json_path  # 图片增强结果保存文件

    # 如果保存文件夹不存在就创建
    if not os.path.exists(save_img_json_path):
        os.mkdir(save_img_json_path)

    for parent, _, files in os.walk(source_img_json_path):
        files.sort()  # 排序一下
        for file in files:
            if file.endswith('jpg') or file.endswith('png') or file.endswith('bmp'):
                cnt = 0
                pic_path = os.path.join(parent, file)
                json_path = os.path.join(parent, file[:-4] + '.json')
                json_dic = toolhelper.parse_json(json_path)
                # 如果图片是有后缀的
                if is_endwidth_dot:
                    # 找到文件的最后名字
                    dot_index = file.rfind('.')
                    _file_prefix = file[:dot_index]  # 文件名的前缀
                    _file_suffix = file[dot_index:]  # 文件名的后缀
                img = cv2.imread(pic_path)

                while cnt < need_aug_num:  # 继续增强
                    auged_img, json_info = dataAug.dataAugment(deepcopy(img), deepcopy(json_dic))
                    img_name = '{}_{}{}'.format(_file_prefix, cnt + 1, _file_suffix)  # 图片保存的信息
                    img_save_path = os.path.join(save_img_json_path, img_name)
                    toolhelper.save_img(img_save_path, auged_img)  # 保存增强图片

                    json_info['imagePath'] = img_name
                    # base64_data = toolhelper.img2str(img_save_path)
                    # json_info['imageData'] = base64_data
                    toolhelper.save_json('{}_{}.json'.format(_file_prefix, cnt + 1),
                                         save_img_json_path, json_info)  # 保存xml文件
                    print(img_name)
                    cnt += 1  # 继续增强下一张
