class DatasetInfo(object):
    dataset_info = {
        'mini_LISAP_train': {
            'image_dir': '../LISAP/mini_Train',
            'anno_dir': '../LISAP/mini_Train_Class_agnostic_annotations.json',
            'split': 'train',
            'weight_file': '../LISAP/weight_data.csv'
        },
        'mini_LISAP_val': {
            'image_dir': '../LISAP/mini_Val',
            'anno_dir': '../LISAP/mini_Val_Class_agnostic_annotations.json',
            'split': 'val',
            'weight_file': '../LISAP/weight_data.csv'
        },
        'mini_LISAP_test': {
            'image_dir': '/data/DATA/PigImageData/mini_Test',
            'anno_dir': '/data/DATA/PigImageData/mini_Test_Class_agnostic_annotations.json',
            'split': 'test',
            'weight_file': '/data/DATA/PigImageData/weight_data.csv'
        },
        'LISAP_test': {
        'image_dir': '../LISAP/Test',
        'anno_dir': '../LISAP/Test_Class_agnostic_annotations.json',
        'split': 'test',
        'weight_file': '../LISAP/weight_data.csv'

        }
    }
