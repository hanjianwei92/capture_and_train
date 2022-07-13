import json
import cv2 as cv
import sys
import numpy as np
import random
import re
import os
import argparse
import multiprocessing
import math
import glob


def read(img_path: str):
    img = cv.imread(img_path)
    try:
        img.shape
    except:
        print('No Such image!---' + str(id) + '.jpg')
        sys.exit(0)
    finally:
        return img

'''
修改：
错误原因：并非修改了初始json文件，而是覆盖了初始json文件
类里增加了一个json_dump_dir，所以现在json_output_dir其实是保存原始json的地方，json_dump_dir是修改后json输出的位置
所以 labelme2coco的labelme_input_dir 应该是 json_dump_dir，这样json错位的问题就解决了
推荐把图像的输入输出路径也分开，这段代码里图像输出到imgs_out文件夹里了，这样原图也不会出错了
'''


class DataAugment(object):
    def __init__(self, src, ori_name: (int or str), img_output_dir, json_output_dir, json_dump_dir):
        # 传入参数值，src是图像，id是名字编号
        self.src = src
        self.id = ori_name
        self.img_output_dir = img_output_dir
        self.json_output_dir = json_output_dir
        self.json_dump_dir = json_dump_dir
        self.height, self.width = src.shape[0:2]
        self.base_img = []

        self.step = 30

    def getname(self, i, op_id):
        if op_id == 1:
            op_name = '_Gaussian'
        elif op_id == 2:
            op_name = '_Epreduce'
        elif op_id == 3:
            op_name = '_Epincrese'
        elif op_id == 4:
            op_name = '_Salt'
        idx = math.ceil(i / 2)
        angle = idx * self.step
        if i == 0:
            name = os.path.join(self.img_output_dir, str(self.id) + op_name + '.jpg')
        elif i % 2 == 1:
            name = os.path.join(self.img_output_dir, str(self.id) + '_rotp' + str(angle) + op_name + '.jpg')
        else:
            name = os.path.join(self.img_output_dir, str(self.id) + '_rotn' + str(angle) + op_name + '.jpg')
        return name

    def new_position(self, angle, n0, m0, mode=0):
        if mode == 1:
            angle = 360 - angle
        a = angle/180 * 3.1416

        N = self.width
        M = self.height

        x1 = n0 - N/2
        y1 = m0 - M/2

        x0 = x1*math.cos(a) + y1*math.sin(a)
        y0 = x1*math.sin(a) + y1*math.cos(a)

        return x0, y0

    def base_img_generate(self):
        for i in range(int(180/self.step+1)):
            if i == 0:
                continue
            angle = self.step * i
            center_x = int(self.width * 0.5)
            center_y = int(self.height * 0.5)
            matRotate_p = cv.getRotationMatrix2D((center_x, center_y), angle, 1)
            matRotate_n = cv.getRotationMatrix2D((center_x, center_y), -angle, 1)
            dst_p = cv.warpAffine(self.src, matRotate_p, dsize=(self.width, self.height))
            dst_n = cv.warpAffine(self.src, matRotate_n, dsize=(self.width, self.height))
            self.base_img.append(dst_p)
            self.base_img.append(dst_n)
            dst_p_path = os.path.join(self.img_output_dir, str(self.id) + '_rotp' + str(angle) + '.jpg')
            if not os.path.exists(dst_p_path): cv.imwrite(dst_p_path, dst_p)
            dst_n_path = os.path.join(self.img_output_dir, str(self.id) + '_rotn' + str(angle) + '.jpg')
            if not os.path.exists(dst_n_path): cv.imwrite(dst_n_path, dst_n)

    def gaussian_blur_fun(self):
        for i in range(len(self.base_img)):
            dst = cv.GaussianBlur(self.base_img[i], (5, 5), 0)
            name = self.getname(i, 1)
            if not os.path.exists(name): cv.imwrite(name, dst)

    def change_exposure_increase(self):
        increase = 1.4
        # brightness
        g = 10
        h, w, ch = self.src.shape
        add = np.zeros([h, w, ch], self.src.dtype)
        for i in range(len(self.base_img)):
            dst = cv.addWeighted(self.base_img[i], increase, add, 1 - increase, g)
            name = self.getname(i, 2)
            if not os.path.exists(name): cv.imwrite(name, dst)

    def change_exposure_reduce(self):
        reduce = 0.5
        # brightness
        g = 10
        h, w, ch = self.src.shape
        add = np.zeros([h, w, ch], self.src.dtype)
        for i in range(len(self.base_img)):
            dst = cv.addWeighted(self.base_img[i], reduce, add, 1 - reduce, g)
            name = self.getname(i, 3)
            if not os.path.exists(name): cv.imwrite(name, dst)

    def add_salt_noise(self):
        percentage = 0.005
        for i in range(len(self.base_img)):
            dst = self.base_img[i]
            num = int(percentage * dst.shape[0] * dst.shape[1])
            for j in range(num):
                rand_x = random.randint(0, dst.shape[0] - 1)
                rand_y = random.randint(0, dst.shape[1] - 1)
                if random.randint(0, 1) == 0:
                    dst[rand_x-1:rand_x+1, rand_y-1:rand_y+1] = 0
                else:
                    dst[rand_x-1:rand_x+1, rand_y-1:rand_y+1] = 256
            name = self.getname(i, 4)
            if not os.path.exists(name): cv.imwrite(name, dst)

    def json_generation(self):
        image_names = glob.glob(os.path.join(self.img_output_dir, str(self.id) + "*.jpg"))
        org_json_path = os.path.join(self.json_output_dir, str(self.id) + ".json")
        image_names = [os.path.split(image_name.split(".")[0])[-1] for image_name in image_names]
        #image_names = ['102_rotp120']      # 这一样行留着当检验用，写图像名字 + 路径
        for image_name in image_names:
            # print(image_name)
            with open(org_json_path, 'r') as js:
                json_data = json.load(js)
                json_data['imageHeight'] = self.height
                json_data['imageWidth'] = self.width
                p = os.path.join(self.img_output_dir, image_name + ".jpg")
                img = cv.imread(p)
                shapes = json_data['shapes']
                # shapes = [shapes[11]]    # 选取诸多类别中的某一个检验其定位情况
                for shape in shapes:
                    points = shape['points']
                    for point in points:
                        match_pattern2 = re.compile(r'(.*)rotp(.*)')
                        match_pattern3 = re.compile(r'(.*)rotn(.*)')
                        b = image_name.split('_')
                        if match_pattern2.match(image_name):      # 检测到正转
                            c = b[1]
                            d = c.split('p')
                            angle = float(d[1])                   # 把角度值拿出来，检测部分暂时不写
                            point[0], point[1] = self.new_position(angle, point[0], point[1], 0)
                        elif match_pattern3.match(image_name):    # 检测到反转
                            c = b[1]
                            d = c.split('n')
                            angle = float(d[1])
                            point[0], point[1] = self.new_position(angle, point[0], point[1], 1)  # 反转模式为 1
                        else:
                            point[0] = point[0]
                            point[1] = point[1]
                json_path = os.path.join(self.json_dump_dir, os.path.split(image_name)[-1] + ".json")
                json_data['imageData'] = None
                real_path = os.path.relpath(os.path.join(self.img_output_dir, image_name + ".jpg"), self.json_output_dir)
                # print(real_path)
                json_data['imagePath'] = real_path
                json.dump(json_data, open(json_path, 'w'), indent=2)
                print("generating", json_path)
        return points, img


def augment(img_path, img_name, img_out_dir, labelme_dir, labelme_dump_dir):
    print("img_path:", img_path, "augmenting")
    img = read(img_path)

    dataAugmentObject = DataAugment(img,
                                    img_name,
                                    img_out_dir,
                                    labelme_dir,
                                    labelme_dump_dir)   # 增强类实例化                                           # 图像扩展
    dataAugmentObject.base_img_generate()
    dataAugmentObject.gaussian_blur_fun()                                           # 进行典型操作
    dataAugmentObject.change_exposure_increase()
    dataAugmentObject.change_exposure_reduce()
    dataAugmentObject.add_salt_noise()
    dataAugmentObject.json_generation()                               # 生成json文件用，不用管返回量，返回量是检验用的


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--labelme_dir", help="input annotated directory", required=True)
    parser.add_argument("--img_dir", help="img directory", required=True)
    parser.add_argument("--img_out_dir", help="img_out_dir directory", required=True)
    parser.add_argument("--labelme_dump_dir", help="labelme_dump_dir directory", required=True)

    args = parser.parse_args()

    img_paths = []
    img_names = []
    for img_path in os.listdir(args.img_dir):
        img_path = os.path.join(args.img_dir, img_path)
        img_name = (os.path.split(img_path)[-1]).split(".")[0]
        ext = (os.path.split(img_path)[-1]).split(".")[1]

        if ext == "png" or ext == "jpg":
            try:
                img_name = int(img_name)
                img_paths.append(img_path)
                img_names.append(img_name)
            except Exception as e:
                print(e)

    cpu_count = int(multiprocessing.cpu_count() / 2) + 1
    # cpu_count = 1
    pool = multiprocessing.Pool(processes=cpu_count)
    for img_path, img_name in zip(img_paths, img_names):
        # pool.apply_async(augment,
        #                  args=(img_path,
        #                        img_name,
        #                        args.img_out_dir,
        #                        args.labelme_dir,
        #                        args.labelme_dump_dir))
        augment(img_path,  img_name, args.img_out_dir, args.labelme_dir, args.labelme_dump_dir)
    pool.close()
    pool.join()

    # for img_path, img_name in zip(img_paths, img_names):
    #     augment(img_path, img_name)










