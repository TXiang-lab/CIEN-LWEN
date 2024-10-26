class DatasetCatalog(object):
    dataset_attrs = {
        'LISAP_train': {
            'data_root': '../LISAP/Train',
            'ann_file': '../LISAP/Train_annotations.json',
            'split': 'train',
            'weight_file': '../LISAP/weight_data.csv'
        },
        'LISAP_val': {
            'data_root': '../LISAP/Val',
            'ann_file': '../LISAP/Val_annotations.json',
            'split': 'val',
            'weight_file': '../LISAP/weight_data.csv'
        },
        'LISAP_test': {
            'data_root': '../LISAP/Test',
            'ann_file': '../LISAP/Test_annotations.json',
            'split': 'test',
            'weight_file': '../LISAP/weight_data.csv'
        }
    }

    @staticmethod
    def get(name):
        attrs = DatasetCatalog.dataset_attrs[name]
        return attrs.copy()
