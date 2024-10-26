import torch
from .utils import decode_ct_hm, clip_to_image, get_gcn_feature,normalize_poly
from .att import ContourSelfAttention

class Refine(torch.nn.Module):
    def __init__(self, c_in=24, num_point=128, stride=4.):
        super(Refine, self).__init__()
        self.num_point = num_point
        self.stride = stride
        self.trans_feature = torch.nn.Sequential(torch.nn.Conv2d(c_in, 256, kernel_size=3,
                                                                 padding=1, bias=True),
                                                 torch.nn.ReLU(inplace=True),
                                                 torch.nn.Conv2d(256, 254, kernel_size=1,
                                                                 stride=1, padding=0, bias=True))
        self.trans_feature2 = torch.nn.Sequential(torch.nn.Conv2d(c_in, 256, kernel_size=3,
                                                                  padding=1, bias=True),
                                                  torch.nn.ReLU(inplace=True),
                                                  torch.nn.Conv2d(256, 254, kernel_size=1,
                                                                  stride=1, padding=0, bias=True))

        self.CSA1 = ContourSelfAttention(256, 8)
        self.CSA2 = ContourSelfAttention(256, 8)

    def csa_1(self, points_features, init_polys, points):
        points_feateres_withpe = torch.cat([points_features, points], dim=-1)

        offsets = self.CSA1(points_feateres_withpe)[:, 1:, :]
        coarse_polys = offsets * self.stride + init_polys.detach()
        return coarse_polys

    def csa_2(self, points_features, coarse_polys, points):
        points_feateres_withpe = torch.cat([points_features, points], dim=-1)

        offsets = self.CSA2(points_feateres_withpe)
        final_polys = offsets + coarse_polys.detach()
        return final_polys

    def forward(self, cnn_feature, ct_polys, init_polys, ct_img_idx, ignore=False):
        # ct 为下采样4倍ct,init_同理
        if ignore or len(init_polys) == 0:
            return init_polys, init_polys
        h, w = cnn_feature.size(2), cnn_feature.size(3)

        feature = self.trans_feature(cnn_feature)
        feature2 = self.trans_feature2(cnn_feature)

        ct_polys = ct_polys.unsqueeze(1).expand(init_polys.size(0), 1, init_polys.size(2))
        points = torch.cat([ct_polys, init_polys], dim=1)
        feature_points = get_gcn_feature(feature, points, ct_img_idx, h, w).permute(0, 2, 1)
        csa1_points = normalize_poly(points)
        coarse_polys = self.csa_1(feature_points, init_polys, csa1_points)

        feature_points = get_gcn_feature(feature2, coarse_polys, ct_img_idx, h, w).permute(0, 2, 1)  ##TODO 修改
        csa2_points = normalize_poly(coarse_polys)
        final_polys = self.csa_2(feature_points, coarse_polys, csa2_points)

        return coarse_polys, final_polys


class Decode(torch.nn.Module):
    def __init__(self, c_in=24, num_point=128, init_stride=10., coarse_stride=4., down_sample=4., min_ct_score=0.05):
        super(Decode, self).__init__()
        self.stride = init_stride
        self.down_sample = down_sample
        self.min_ct_score = min_ct_score
        self.refine = Refine(c_in=c_in, num_point=num_point, stride=coarse_stride)

    def train_decode(self, data_input, output, cnn_feature):
        wh_pred = output['wh']
        ct_01 = data_input['ct_01'].bool()
        ct_ind = data_input['ct_ind'][ct_01]
        ct_img_idx = data_input['ct_img_idx'][ct_01]
        _, _, height, width = data_input['ct_hm'].size()
        ct_x, ct_y = ct_ind % width, ct_ind // width

        if ct_x.size(0) == 0:
            ct_offset = wh_pred[ct_img_idx, :, ct_y, ct_x].view(ct_x.size(0), 1, 2)
        else:
            ct_offset = wh_pred[ct_img_idx, :, ct_y, ct_x].view(ct_x.size(0), -1, 2)

        ct_x, ct_y = ct_x[:, None].to(torch.float32), ct_y[:, None].to(torch.float32)  # gt
        ct = torch.cat([ct_x, ct_y], dim=1)  # gt

        init_polys = ct_offset * self.stride + ct.unsqueeze(1).expand(ct_offset.size(0),
                                                                      ct_offset.size(1), ct_offset.size(2))
        coarse_polys, final_polys = self.refine(cnn_feature, ct, init_polys, ct_img_idx.clone())

        output.update({'poly_init': init_polys * self.down_sample})
        output.update({'poly_coarse': coarse_polys * self.down_sample})
        output.update({'poly_final': final_polys * self.down_sample})
        return

    def test_decode(self, cnn_feature, output, K=100, min_ct_score=0.05, ignore=False):
        hm_pred, wh_pred = output['ct_hm'], output['wh']
        poly_init, detection = decode_ct_hm(torch.sigmoid(hm_pred), wh_pred,
                                            K=K, stride=self.stride)
        valid = detection[0, :, 2] >= min_ct_score
        poly_init, detection = poly_init[0][valid], detection[0][valid]

        init_polys = clip_to_image(poly_init, cnn_feature.size(2), cnn_feature.size(3))
        output.update({'poly_init': init_polys * self.down_sample})

        img_id = torch.zeros((len(poly_init),), dtype=torch.int64)
        poly_coarse, poly_final = self.refine(cnn_feature, detection[:, :2], poly_init, img_id,
                                                        ignore=ignore)
        coarse_polys = clip_to_image(poly_coarse, cnn_feature.size(2), cnn_feature.size(3))
        final_polys = clip_to_image(poly_final, cnn_feature.size(2), cnn_feature.size(3))
        output.update({'poly_coarse': coarse_polys * self.down_sample})
        output.update({'poly_final': final_polys * self.down_sample})
        output.update({'detection': detection})
        return

    def forward(self, data_input, cnn_feature, output=None, is_training=True, ignore=False):
        if is_training:
            self.train_decode(data_input, output, cnn_feature)
        else:
            self.test_decode(cnn_feature, output, min_ct_score=self.min_ct_score,
                             ignore=ignore)
