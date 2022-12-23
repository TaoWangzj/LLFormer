import os
from transform.dataset_RGB import DataLoaderTrain, DataLoaderVal, DataLoaderTest_,DataLoaderVal_


def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options)


def get_validation_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, img_options)

def get_validation_data2(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal_(rgb_dir, img_options)


def get_test_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest_(rgb_dir, img_options)
