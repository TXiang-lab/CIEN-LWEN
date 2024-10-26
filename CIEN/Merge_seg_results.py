import json

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import pandas as pd
import datetime
from dataset.info import DatasetInfo
import os


def merge_id(cfg):
    info = DatasetInfo.dataset_info[cfg.test.dataset]
    GT_ANN_JSON = info['image_dir'] + '_annotations.json'
    SEG_RESULT_JSON = os.path.join(cfg.commen.result_dir, 'results.json')
    PY_RESULT_JSON = os.path.join(cfg.commen.result_dir, 'py_results.json')
    SAVE_SEG_JSON = os.path.join(cfg.commen.result_dir, 'img_results_with_ID_and_Weight.json')
    SAVE_PY_JSON = os.path.join(cfg.commen.result_dir, 'py_results_with_ID_and_Weight.json')
    weight_file = info['weight_file']
    weight_data = pd.read_csv(weight_file, header=0, index_col=0)
    weight_data.columns = [datetime.datetime.strptime(x, '%Y/%m/%d').date() for x in weight_data.columns]

    def get_ind(dt_anns, gt_anns):
        d = [i['segmentation'] for i in dt_anns]
        g = [i['segmentation'] for i in gt_anns]
        g_amount = len(g)
        crowd = [0 for _ in range(g_amount)]
        ious = maskUtils.iou(d, g, crowd)

        match_ind = {}
        for i in range(g_amount):
            matched = np.argsort(ious[:, i])[::-1]
            for m in matched:
                if m not in match_ind.keys():
                    match_ind[m] = i
                    break
        return match_ind

    def merge_dic(data, dic, gt):
        cache = []
        for d_id, g_id in dic.items():
            annotation = data[d_id - 1]
            cat = gt.loadAnns(g_id)[0]['category_id']
            if cat == 40:
                continue
            annotation['category_id'] = cat
            cache.append(annotation)
        return cache

    def get_weight_filename(wd, gt, img_id, cls_id):
        filename = gt.loadImgs(img_id)[0]['file_name']
        date = '/'.join(filename.split('_')[3:6])
        date = datetime.datetime.strptime(date, '%Y/%m/%d').date()

        weight = float(wd.loc[cls_id, date])

        return weight, filename

    GT = COCO(GT_ANN_JSON)
    dets = GT.loadRes(SEG_RESULT_JSON)
    coco_eval = COCOeval(GT, dets, 'segm')
    coco_eval.params.useCats = 0
    coco_eval.evaluate()

    dt_cats_dic = {}
    for img_id in GT.getImgIds():

        g_ids = GT.getAnnIds(img_id)
        d_ids = dets.getAnnIds(img_id)
        if not d_ids:
            continue

        g_anns = GT.loadAnns(g_ids)
        d_anns = dets.loadAnns(d_ids)

        m_dic = get_ind(d_anns, g_anns)

        for d_index, g_index in m_dic.items():
            dt_cats_dic[d_ids[d_index]] = g_ids[g_index]

    py_data = json.load(open(PY_RESULT_JSON, 'r'))
    seg_data = json.load(open(SEG_RESULT_JSON, 'r'))

    pys = merge_dic(py_data, dt_cats_dic, GT)
    segs = merge_dic(seg_data, dt_cats_dic, GT)

    for i in pys:
        w, fn = get_weight_filename(weight_data, GT, i['image_id'], i['category_id'])
        i['weight'] = w
        i['file_name'] = fn

    for i in segs:
        w, fn = get_weight_filename(weight_data, GT, i['image_id'], i['category_id'])
        i['weight'] = w
        i['file_name'] = fn
    with open(SAVE_PY_JSON, 'w') as f:
        json.dump(pys, f)
    with open(SAVE_SEG_JSON, 'w') as f:
        json.dump(segs, f)
