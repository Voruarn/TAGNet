"""
@Project: PICR_Net
@File: setting/dataloader.py
@Author: chen zhang
@Institution: Beijing JiaoTong University
"""

import os
import random
import numpy as np
from PIL import Image, ImageEnhance
from torchvision import transforms
from torch.utils import data


class SalObjDataset(data.Dataset):

    def __init__(self, image_root, depth_root, gt_root, text_root, trainsize):
        """
        :param image_root: The path of RGB training images.
        :param depth_root: The path of depth training images.
        :param gt_root: The path of training ground truth.
        :param text_root: The path of text descriptions for training images.
        :param trainsize: The size of training images.
        """
        self.trainsize = trainsize
        # 加载图像路径
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.texts = [text_root + f for f in os.listdir(text_root) if f.endswith('.txt')]
    
        # 排序确保数据对应
        self.images = sorted(self.images)
        self.depths = sorted(self.depths)
        self.gts = sorted(self.gts)
        self.texts = sorted(self.texts)
        
        # 过滤不匹配的文件
        self.filter_files()
        self.size = len(self.images)
        
        # 数据转换
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.depths_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, ], [0.229, ])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        # 加载图像数据
        image = self.rgb_loader(self.images[index])
        depth = self.binary_loader(self.depths[index])
        gt = self.binary_loader(self.gts[index])
        
        # 加载文本数据
        text = self.text_loader(self.texts[index])
        
        # 数据增强
        image, depth, gt = randomFlip(image, depth, gt)
        image, depth, gt = randomRotation(image, depth, gt)
        
        # 数据转换
        image = self.img_transform(image)
        depth = self.depths_transform(depth)
        gt = self.gt_transform(gt)
        
        return image, depth, text, gt

    def __len__(self):
        return self.size

    def filter_files(self):
        """ Check whether a set of images match in size. """
        assert len(self.images) == len(self.depths) == len(self.gts) == len(self.texts)
        images = []
        depths = []
        gts = []
        texts = []
        
        for img_path, depth_path, gt_path, text_path in zip(self.images, self.depths, self.gts, self.texts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                depths.append(depth_path)
                gts.append(gt_path)
                texts.append(text_path)
            else:
                print(f"Warning: Image sizes do not match - {img_path} and {gt_path}, skipping...")
        
        self.images = images
        self.depths = depths
        self.gts = gts
        self.texts = texts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            # Removing alpha channel.
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')
    
    def text_loader(self, path):
        """读取文本文件内容"""
        with open(path, 'r', encoding='utf-8') as f:
            # 读取文本并去除首尾空白字符
            return f.read().strip()


def randomFlip(img, depth, gt):
    flip_flag = random.randint(0, 2)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip_flag == 2:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
        gt = gt.transpose(Image.FLIP_TOP_BOTTOM)
    return img, depth, gt


def randomRotation(image, depth, gt):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        depth = depth.rotate(random_angle, mode)
        gt = gt.rotate(random_angle, mode)
    return image, depth, gt


def get_loader(image_root, depth_root, gt_root, text_root, batchsize, trainsize, 
               shuffle=True, num_workers=4, pin_memory=True):
    dataset = SalObjDataset(image_root, depth_root, gt_root, text_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader, dataset.size


class test_dataset:

    def __init__(self, image_root, depth_root, gt_root, text_root, testsize):
        """
        :param image_root: The path of RGB testing images.
        :param depth_root: The path of depth testing images.
        :param gt_root: The path of testing ground truth.
        :param text_root: The path of text descriptions for testing images.
        :param testsize: The size of testing images.
        """
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.depth = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.jpg') or f.endswith('.png')
                      or f.endswith('.bmp')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.texts = [text_root + f for f in os.listdir(text_root) if f.endswith('.txt')]
    
        # 排序确保数据对应
        self.images = sorted(self.images)
        self.depth = sorted(self.depth)
        self.gts = sorted(self.gts)
        self.texts = sorted(self.texts)
        
        # 数据转换
        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.depth_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, ], [0.229, ])
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def __len__(self):
        return self.size

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.img_transform(image).unsqueeze(0)
        
        depth = self.binary_loader(self.depth[self.index])
        depth = self.depth_transform(depth).unsqueeze(0)
        
        gt = self.binary_loader(self.gts[self.index])
        
        # 加载文本数据
        text = self.text_loader(self.texts[self.index])
        
        name = self.images[self.index].split('\\')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        
        self.index += 1
        self.index = self.index % self.size
        
        return image, depth, text, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')
    
    def text_loader(self, path):
        """读取文本文件内容"""
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
