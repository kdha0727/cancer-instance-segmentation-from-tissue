import numpy as np
import os
from torch.utils.data import IterableDataset
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToTensor

from .numpy import NumpyDataset


class LazyNumpyDataset(IterableDataset, NumpyDataset):

    __length_cache = None

    def __init__(
            self,
            image_path,
            mask_path,
            type_path=None,
            transform=ToTensor(),
            target_transform=ToTensor(),
            sampler_class=None,
            sampler_kwargs=None,
            length=None,
    ):
        VisionDataset.__init__(
            self,
            root=None,  # type: ignore
            transforms=None,
            transform=transform,
            target_transform=target_transform,
        )

        self.image_path = image_path
        assert os.path.isfile(image_path)
        self.mask_path = mask_path
        assert os.path.isfile(mask_path)
        self.type_path = type_path
        assert type_path is None or os.path.isfile(type_path)
        if length is not None:
            self.__length_cache = length

        if sampler_class is not None:
            self.sampler = sampler_class(self, **(sampler_kwargs or {}))
        else:
            self.sampler = None

    def __len__(self):
        if self.__length_cache is None:
            self.__length_cache = np.load(self.image_path).shape[0]
        return self.__length_cache

    def __iter__(self):
        image_array = np.load(self.image_path)
        mask_array = np.load(self.mask_path)
        if self.__length_cache is None or self.__length_cache != image_array.shape[0]:
            self.__length_cache = image_array.shape[0]
        sampler = self.sampler or range(len(image_array))
        for index in sampler:
            image, mask = image_array[index], mask_array[index]
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                mask = self.target_transform(mask)
            yield image, mask

    def __getitem__(self, item):
        raise NotImplementedError
