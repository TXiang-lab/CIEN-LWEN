from .coco import Evaluator, DetectionEvaluator
from dataset.info import DatasetInfo


def _evaluator_factory(result_dir, anno_file, eval_format, cfg):
    if eval_format == 'segm':
        evaluator = Evaluator(result_dir, anno_file, cfg)
    else:
        evaluator = DetectionEvaluator(result_dir, anno_file, cfg)
    return evaluator


def make_evaluator(cfg):
    anno_file = DatasetInfo.dataset_info[cfg.test.dataset]['anno_dir']
    eval_format = cfg.test.segm_or_bbox
    return _evaluator_factory(cfg.commen.result_dir, anno_file, eval_format, cfg)
