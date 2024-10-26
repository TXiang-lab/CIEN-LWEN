from .weight_evaluate import Evaluator


def make_evaluator(cfg):
    if cfg.skip_eval:
        return None
    else:
        return Evaluator(cfg)