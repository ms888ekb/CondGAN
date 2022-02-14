from torch.utils import data
import os
import numpy as np
from glob import glob
from osgeo import gdal, gdalconst
import cv2


class GeoDataLoader(data.Dataset):
    def __init__(self, x_data=None, y_data=None, resize=None,
                 mean=None, std=None, shuffle=True):
        if y_data is None:
            self.iamsource = False
        else:
            self.iamsource = True
        self.source_data = self.__lookup(x_data)
        if y_data is not None:
            self.source_labels = y_data
        self.src_indexes = []
        self.resize = resize
        self.shuffle = shuffle
        self.dataset_size = None
        self.mean = 127.5 if mean is None else mean
        self.std = 127 if std is None else std
        self.__on_init()

    def get_sample_batch(self, num=1):
        sample_indexes = np.random.randint(0, high=len(self), size=num)
        return self[sample_indexes[0]]

    def get_num_samples(self):
        return self.dataset_size

    def __on_init(self):
        source_data_index = np.arange(len(self.source_data))
        if self.shuffle:
            np.random.shuffle(source_data_index)
        self.dataset_size = len(source_data_index)
        self.src_indexes = source_data_index

    @staticmethod
    def __lookup(path):
        mask = os.path.join(path, "*.TIF")
        f_list = glob(mask)
        return f_list

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        index = self.src_indexes[index]
        image_ds = gdal.Open(self.source_data[index], gdalconst.GA_ReadOnly)
        image_georef = image_ds.GetGeoTransform()
        assert image_ds is not None, "The input image is None. Check the image path."
        image = image_ds.ReadAsArray()
        assert image.shape[0] == 4, "Wrong input image band number"
        image = np.asarray(image, np.float)
        image = image.transpose((1, 2, 0))
        image = cv2.resize(image, dsize=self.resize, interpolation=cv2.INTER_CUBIC)
        image -= self.mean
        image /= self.std
        image = image.transpose((2, 0, 1))
        image = image[:3, :, :]

        sample_name = os.path.split(self.source_data[index])[1]
        if not os.path.exists(os.path.join(self.source_labels, sample_name)):
            print(f'Corresponding label for {sample_name} was not found.')
        label_ds = gdal.Open(os.path.join(self.source_labels, sample_name), gdalconst.GA_ReadOnly)
        label = label_ds.ReadAsArray()
        label = np.asarray(label, np.uint8)
        label = cv2.resize(label, dsize=self.resize, interpolation=cv2.INTER_NEAREST)
        return image.copy(), label.copy(), image_georef
