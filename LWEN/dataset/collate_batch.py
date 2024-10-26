from torch.utils.data.dataloader import default_collate


def my_collator(batch):
    try:
        imgs = default_collate([b[0] for b in batch if b[0] is not None])
        polys = default_collate([b[1] for b in batch if b[1] is not None])
        input_polys = default_collate([b[2] for b in batch if b[2] is not None])
        wts = default_collate([b[3] for b in batch if b[3] is not None])
        img_ids = default_collate([b[4] for b in batch if b[4] is not None])
        cls_ids = default_collate([b[5] for b in batch if b[5] is not None])
        ret = {'img': imgs, 'poly': polys, 'norm_poly': input_polys, 'weight': wts, 'img_id': img_ids,
               'cls_id': cls_ids}
        return ret
    except:
        return None
