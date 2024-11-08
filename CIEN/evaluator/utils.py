import pycocotools.mask as mask_utils
import numpy as np
import cv2
from scipy.spatial.distance import cdist


def point_sort(poly):
    assert poly.ndim == 2
    best_path = []
    best_dis = 9999
    dis = cdist(poly, poly)
    p_num = len(poly)
    for start_point in range(p_num):
        path_length = []
        cache = [start_point]
        while len(cache) != p_num:
            i = cache[-1]
            s = np.argsort(dis[i])
            s = s[s != i]
            for id in s:
                if id not in cache:
                    cache.append(id)
                    path_length.append(dis[i][id])
                    break
        if np.mean(path_length) < best_dis:
            best_path = cache[:]
            best_dis = np.mean(path_length)

    return poly[best_path]


# def point_sort(poly):
#     assert poly.ndim == 2
#
#     cache = [0]
#     dis = cdist(poly, poly)
#     while len(cache) != len(poly):
#         i = cache[-1]
#         s = np.argsort(dis[i])
#         s = s[s != i]
#         for id in s:
#             if id not in cache:
#                 cache.append(id)
#                 break
#     return poly[cache]


def poly_sort_coco_poly_to_rle(poly, h, w):
    rle_ = []
    for i in range(len(poly)):
        sorted_poly = point_sort(poly[i])
        rles = mask_utils.frPyObjects([sorted_poly.reshape(-1)], h, w)
        rle = mask_utils.merge(rles)
        rle['counts'] = rle['counts'].decode('utf-8')
        rle_.append(rle)
    return rle_


def coco_poly_to_rle(poly, h, w):
    rle_ = []
    for i in range(len(poly)):
        rles = mask_utils.frPyObjects([poly[i].reshape(-1)], h, w)
        rle = mask_utils.merge(rles)
        rle['counts'] = rle['counts'].decode('utf-8')
        rle_.append(rle)
    return rle_


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    """pt: [n, 2]"""
    new_pt = np.dot(np.array(pt), t[:, :2].T) + t[:, 2]
    return new_pt
