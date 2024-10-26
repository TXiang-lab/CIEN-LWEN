import os
import json
import numpy as np
from sklearn.metrics import r2_score
from dataset.dataset_catalog import DatasetCatalog
import pycocotools.coco as coco
from .process_result_dict import initial_dict, caculate_mean, draw_stat_fig


class Evaluator:
    def __init__(self, cfg):
        self.results = []
        args = DatasetCatalog.get(cfg.val.dataset)
        self.coco = coco.COCO(args['ann_file'])
        self.keys = list(self.coco.catToImgs.keys())
        self.data = initial_dict(self.keys)

        self.result_dir = cfg.result_dir
        os.system('mkdir -p {}'.format(self.result_dir))

    def evaluate(self, output, batch):
        predict_weights = output['predict_weights'].detach().cpu().numpy()
        true_weights = output['true_weights'].detach().cpu().numpy()
        catids = output['catids'].detach().cpu().numpy()

        img_id = int(batch['img_id'][0])
        name = self.coco.loadImgs(img_id)[0]['file_name']
        date = '/'.join(name.split('_')[4:6])

        weight_dets = []
        for i in range(predict_weights.shape[0]):
            weight_eval = {
                'image_id': img_id,
                'date': date,
                'category_id': int(catids[i][0]),
                'predict_weight': float(predict_weights[i][0]),
                'true_weight': float(true_weights[i][0]),
                'abs_bias': float(np.abs(predict_weights[i][0] - true_weights[i][0]))
            }
            weight_dets.append(weight_eval)

            self.data[str(catids[i][0])][date]['predict'].append(float(predict_weights[i][0]))
            if 'true' not in self.data[str(catids[i][0])][date]:
                self.data[str(catids[i][0])][date]['true'] = float(true_weights[i][0])

        self.results.extend(weight_dets)

    def summarize(self):
        json.dump(self.results, open(os.path.join(self.result_dir, 'results.json'), 'w'))
        self.results = []

        global_stat = caculate_mean(self.data)
        json.dump(self.data, open(os.path.join(self.result_dir, 'stat_results.json'), 'w'))
        json.dump(global_stat, open(os.path.join(self.result_dir, 'global_stat_results.json'), 'w'))

        D = json.load(open(os.path.join(self.result_dir, 'stat_results.json'), 'r'))
        p = np.array([result['predict'] for PigId, day in D.items() for d, result in day.items()])
        t = np.array([result['true'] for PigId, day in D.items() for d, result in day.items()])
        mae = np.mean(np.abs(p - t))
        mse = np.mean(np.square(p - t))
        mape = np.mean(np.abs(p - t) / t)
        r2 = r2_score(t, p)
        print('MAE:{:.4f} , MSE:{:.4f} , MAPE:{:.4f}, R2:{:.4f}'.format(mae, mse, mape * 100, r2))
        fig_stat = {}
        for id, data in self.data.items():
            fig_stat[id] = draw_stat_fig(data, 'PIG{}'.format(id))
        fig_stat['Global'] = draw_stat_fig(global_stat, 'ALL')

        self.data = initial_dict(self.keys)

        return {'MAE': mae, 'MSE': mse, 'MAPE': mape * 100, 'R2': r2}, fig_stat
