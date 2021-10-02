import numpy as np
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToTensor


class NumpyDataset(VisionDataset):  # 메모리 충분하신 분만 쓰세요

    def __init__(
            self,
            image_path,
            mask_path,
            type_path=None,
            transform=ToTensor(),
            target_transform=ToTensor(),
    ):
        VisionDataset.__init__(
            self,
            root=None,  # type: ignore
            transforms=None,
            transform=transform,
            target_transform=target_transform,
        )

        self.image_array = np.load(image_path)
        self.mask_array = np.load(mask_path)
        assert self.image_array.shape[0] == self.mask_array.shape[0]
        if type_path is not None:
            self.type_array = np.load(type_path)
            assert self.image_array.shape[0] == self.type_array.shape[0]
        else:
            self.type_array = None

    def __len__(self):
        return self.image_array.shape[0]

    def __iter__(self):
        for index in range(len(self)):
            image, mask = self.image_array[index], self.mask_array[index]
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                mask = self.target_transform(mask)
            yield image, mask

    def __getitem__(self, index):
        image, mask = self.image_array[index], self.mask_array[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return image, mask
