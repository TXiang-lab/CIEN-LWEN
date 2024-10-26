from termcolor import colored
import torch
def load_network(net, checkpoint, resume=True, strict=True):
    if not resume:
        return 0

    if not os.path.exists(checkpoint):
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0

    print('load model: {}'.format(checkpoint))
    pretrained_model = torch.load(checkpoint)
    net.load_state_dict(pretrained_model['net'], strict=strict)
    return pretrained_model['epoch'] + 1

import os
import cv2
import numpy as np

import torch.utils.data as data
import torch
from dataset.utils import augment, get_affine_transform
import json
from pycocotools import mask as maskUtils
from dataset.dataset_catalog import DatasetCatalog
from dataset.collate_batch import my_collator


def random_rod(img):
    inp = img.astype(dtype=np.float32)
    # rod = np.random.randint(0, 180)
    rod = int(90)
    center = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    scale = max(img.shape[0], img.shape[1]) * 1.0
    input_h, input_w = img.shape[0], img.shape[1]
    trans_input = get_affine_transform(center, scale, rod, [input_w, input_h])
    inp = cv2.warpAffine(inp, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
    return inp


def get_cropped(img):
    index = np.where(img != 0)
    min_0 = min(index[0])
    max_0 = max(index[0])
    min_1 = min(index[1])
    max_1 = max(index[1])
    cropped = img[min_0:max_0 + 1, min_1:max_1 + 1, :]
    return cropped


def get_shifted(img):
    cropped = get_cropped(img)
    shifted = np.zeros_like(img)
    shifted[:cropped.shape[0], :cropped.shape[1], :] = cropped
    return shifted


class Poly_Dataset(data.Dataset):
    def __init__(self, data_root, results_file):
        super(Poly_Dataset, self).__init__()

        self.data_root = data_root

        self.anns = json.load(open(results_file, 'r'))  # shape(5623,)

    def filter_boundary_polys(self, p, h=576, w=768):
        if (p[:, 0] < w - 5).all() and (p[:, 1] < h - 5).all():
            return p
        return None

    def normlize_poly(self, poly, h=576, w=768):
        img_h, img_w = h, w
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

    def __getitem__(self, index):
        ann = self.anns[index]  # 1

        poly = np.array(ann['segmentation'])
        poly = self.filter_boundary_polys(poly)
        if poly is None:
            return None, None, None, None, None, None
        input_poly = self.normlize_poly(poly)
        return torch.FloatTensor(0), torch.FloatTensor(poly), torch.FloatTensor(
            input_poly), torch.FloatTensor(
            [ann['weight']]), torch.IntTensor([ann['image_id']]), torch.IntTensor([ann['category_id']])

    def __len__(self):
        return len(self.anns)


class Img_Dataset(data.Dataset):
    def __init__(self, data_root, results_file, process='ORIGIN', output_mask=True):
        super(Img_Dataset, self).__init__()

        self.data_root = data_root

        self.anns = json.load(open(results_file, 'r'))  # shape(5623,)
        self.output_mask = output_mask
        self.process = process

    def filter_boundary_mask(self, m, h=576, w=768):
        index = np.where(m != 0)
        if (index[1] < w - 5).all() and (index[0] < h - 5).all():
            return m
        return None

    def norm_img(self, img, mask):
        orig_img, inp, trans_input, center, scale, inp_out_hw, flipped, width = augment(
            img, 'test'
        )

        inp = inp.transpose(1, 2, 0)
        masked = cv2.bitwise_and(inp, inp, mask=mask)
        if self.output_mask:
            masked = np.where(masked != 0, 1, 0)  # TODO: only use mask
        return masked

    def __getitem__(self, index):
        ann = self.anns[index]  # 1

        mask = maskUtils.decode(ann['segmentation'])
        mask = self.filter_boundary_mask(mask)
        if mask is None:
            return None, None, None, None, None, None
        img = cv2.imread(os.path.join(self.data_root, ann['file_name']))
        input_img = self.norm_img(img, mask)

        if self.process == "FLIP":
            input_img = input_img[:, ::-1, :].copy()

        if self.process == 'ROD':
            input_img = random_rod(input_img)
            input_img = self.filter_boundary_mask(input_img)
            if input_img is None:
                return None, None, None, None, None, None

        if self.process == 'CROP':
            input_img = get_cropped(input_img)

        if self.process == 'SHIFT':
            input_img = get_shifted(input_img)

        return torch.FloatTensor(input_img.transpose(2, 0, 1)), torch.FloatTensor(0), torch.FloatTensor(
            0), torch.FloatTensor(
            [ann['weight']]), torch.IntTensor([ann['image_id']]), torch.IntTensor([ann['category_id']])

    def __len__(self):
        return len(self.anns)


def make_data_loader(args):
    data_path = DatasetCatalog.get(args.dataset)['data_root']
    assert args.process in ['ORIGIN', 'FLIP', 'ROD', 'CROP', 'SHIFT']
    if args.seg_mode == 'PY':
        dataset = Poly_Dataset(data_path, args.seg_results)
    elif args.seg_mode == 'MSK':
        dataset = Img_Dataset(data_path, args.seg_results, process=args.process)
    else:
        dataset = Img_Dataset(data_path, args.seg_results, process=args.process, output_mask=False)

    data_loader = torch.utils.data.DataLoader(dataset, num_workers=64, collate_fn=my_collator)
    return data_loader
