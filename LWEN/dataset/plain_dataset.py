import os
import cv2
import numpy as np

import torch.utils.data as data
from pycocotools.coco import COCO
import pandas as pd
import datetime
import torch
from .utils import augment, uniformsample, filter_tiny_polys, get_cw_polys, transform_polys


def random_sample(p, t, p_num=128, has_repeat=False):
    up = uniformsample(p, p_num * t)
    index = np.random.choice([i for i in range(p_num * t)], p_num, replace=has_repeat)
    index.sort()
    return up[index, :]


class Dataset(data.Dataset):
    def __init__(self, ann_file, data_root, split, weight_file):
        super(Dataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.coco = COCO(ann_file)
        self.anns = np.array(sorted(self.coco.getAnnIds()))
        self.weight_data = pd.read_csv(weight_file, header=0, index_col=0)
        self.weight_data.columns = [datetime.datetime.strptime(x, '%Y/%m/%d').date() for x in self.weight_data.columns]
        self.get_bf_ann()

    def get_bf_ann(self):
        dates = self.weight_data.columns.values
        anns = []
        for i in self.anns:
            anno = self.coco.loadAnns(int(i))
            if int(anno[0]['category_id']) == 40:
                continue
            img_id = anno[0]['image_id']
            filename = self.coco.loadImgs(int(img_id))[0]['file_name']
            date = '/'.join(filename.split('_')[3:6])
            date = datetime.datetime.strptime(date, '%Y/%m/%d').date()
            if date in dates:
                anns.append(i)
        self.anns = anns

    def process_info(self, ann_id):
        anno = self.coco.loadAnns(int(ann_id))
        img_id = anno[0]['image_id']
        filename = self.coco.loadImgs(int(img_id))[0]['file_name']
        cls_id = anno[0]['category_id']
        return anno, filename, img_id, cls_id

    def filter_boundary_polys(self, polys, h, w):
        return [poly for poly in polys if (poly[:, 0] < w - 5).all() and (poly[:, 1] < h - 5).all()]

    def get_valid_polys(self, instance_poly, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        if len(instance_poly) < 4:
            return None

        instance_poly[:, 0] = np.clip(instance_poly[:, 0], 0, output_w - 1)
        instance_poly[:, 1] = np.clip(instance_poly[:, 1], 0, output_h - 1)
        polys = filter_tiny_polys([instance_poly])
        polys = self.filter_boundary_polys(polys, output_h, output_w)
        polys = get_cw_polys(polys)
        polys = [poly[np.sort(np.unique(poly, axis=0, return_index=True)[1])] for poly in polys]
        if len(polys) == 0:
            return None
        return polys[0]

    def transform_original_data(self, poly, trans_output, inp_out_hw, flipped, width):
        output_h, output_w = inp_out_hw[2:]

        if flipped:
            poly[:, 0] = width - np.array(poly[:, 0]) - 1

        poly = transform_polys([poly], trans_output, output_h, output_w)[0]
        poly = self.get_valid_polys(poly, inp_out_hw)
        if poly is None:
            return None
        if self.split == 'train':
            sampled_poly = random_sample(poly, 4)
        else:
            sampled_poly = uniformsample(poly, 128)
        return sampled_poly

    def augment_poly_img(self, poly, img):
        orig_img, inp, trans_input, center, scale, inp_out_hw, flipped, width = augment(
            img, self.split
        )
        poly_ = self.transform_original_data(poly, trans_input, inp_out_hw, flipped, width)

        if poly_ is None:
            return None, None

        inp = inp.transpose(1, 2, 0)
        mask = np.zeros(inp.shape[:2], dtype="uint8")
        cache = poly_.reshape(1, -1, 2).astype('int32')
        _ = cv2.polylines(mask, cache, 1, 255)
        _ = cv2.fillPoly(mask, cache, 255)
        masked = cv2.bitwise_and(inp, inp, mask=mask)
        return poly_, masked

    def read_auged_data(self, anno, filename):
        img = cv2.imread(os.path.join(self.data_root, filename))
        poly = np.array(anno[0]['segmentation'], dtype=np.int32).reshape(-1, 2)
        instance_poly, masked = self.augment_poly_img(poly, img)
        if instance_poly is None:
            return None, None
        return masked, instance_poly

    def process_weight(self, cls_id, filename):
        date = '/'.join(filename.split('_')[3:6])
        date = datetime.datetime.strptime(date, '%Y/%m/%d').date()
        try:
            weight = float(self.weight_data.loc[cls_id, date])
        except:
            return None
        return weight

    def normlize_poly(self, poly, img):
        img_h, img_w = img.shape[0], img.shape[1]
        img_norm_poly, box_norm_poly = np.zeros_like(poly), np.zeros_like(poly)
        img_norm_poly[:, 0] = poly[:, 0] / img_w
        img_norm_poly[:, 1] = poly[:, 1] / img_h
        ##计算box归一化坐标
        min_x, min_y = np.min(poly[:, 0]), np.min(poly[:, 1])
        max_x, max_y = np.max(poly[:, 0]), np.max(poly[:, 1])
        len_x, len_y = max_x - min_x, max_y - min_y
        box_norm_poly[:, 0] = (poly[:, 0] - min_x) / len_x
        box_norm_poly[:, 1] = (poly[:, 1] - min_y) / len_y
        return np.concatenate([img_norm_poly, box_norm_poly], axis=-1)

    def add_noise(self, p):
        if self.split != 'train':
            return p
        rand_noise = np.random.rand(p.shape[0], p.shape[1])
        rand_noise = rand_noise * 2
        return p + rand_noise

    def __getitem__(self, index):
        ann = self.anns[index]  # 1

        anno, filename, img_id, cls_id = self.process_info(ann)  # anno:list contains{},filename:str pic_name,
        wt = self.process_weight(cls_id, filename)
        img, poly = self.read_auged_data(anno, filename)

        if img is None:
            return None, None, None, None, None, None

        poly = self.add_noise(poly)
        input_poly = self.normlize_poly(poly, img)
        return torch.FloatTensor(img.transpose(2, 0, 1)), torch.FloatTensor(poly), torch.FloatTensor(
            input_poly), torch.FloatTensor(
            [wt]), torch.IntTensor([img_id]), torch.IntTensor([cls_id])

    def __len__(self):
        return len(self.anns)
