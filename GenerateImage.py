# encoding: utf-8
import random

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw

# 数据集，可根据需要增加英文或其它字符
DIGITS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
          "W", "X", "Y", "Z"]

# 数据集目录、模型目录
data_dir = 'D:/tmp/lstm_ctc_data2/'


# 生成椒盐噪声
def img_salt_pepper_noise(src, percentage):
    noise_img = src
    noise_num = int(percentage * src.shape[0] * src.shape[1])
    for i in range(noise_num):
        rand_x = random.randint(0, src.shape[0] - 1)
        rand_y = random.randint(0, src.shape[1] - 1)
        if random.randint(0, 1) == 0:
            noise_img[rand_x, rand_y] = 0
        else:
            noise_img[rand_x, rand_y] = 255
    return noise_img


# 随机生成不定长标签的图片集
def gen_text(cnt):
    # 设置文字字体和大小
    font_path = r'D:/arial.ttf'
    font_size = 30
    font = ImageFont.truetype(font_path, font_size)

    for i in range(cnt):
        # 随机生成1到10位的不定长数字
        rnd = random.randint(1, 10)
        text = ''
        for j in range(rnd):
            text = text + DIGITS[random.randint(0, len(DIGITS) - 1)]

        # 生成图片并绘上文字
        img = Image.new("RGB", (256, 32))
        draw = ImageDraw.Draw(img)
        draw.text((1, 1), text, font=font, fill='white')
        img = np.array(img)

        # 随机叠加椒盐噪声并保存图像
        img = img_salt_pepper_noise(img, float(random.randint(1, 10) / 100.0))
        img_path = data_dir + text + '.jpg'
        cv2.imwrite(img_path, img)
        print("生成图片：", img_path)


gen_text(5000)
