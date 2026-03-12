import h5py
import torch
import json
import numpy as np
import cv2
import os
import kagglehub
import torchvision.transforms as transforms


class DatasemyYaleCroppedB(torch.utils.data.Dataset):
    def __init__(self, json_path):
        self.pictures_base_path = kagglehub.dataset_download(
            "jensdhondt/extendedyaleb-cropped-full")
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        with open(json_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(
            self.pictures_base_path, self.data[index]['img']))
        tmplt = cv2.imread(os.path.join(
            self.pictures_base_path, self.data[index]['tmplt']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tmplt = cv2.cvtColor(tmplt, cv2.COLOR_BGR2GRAY)

        init = torch.tensor(self.data[index]['p_init']).float()

        A = np.array(self.data[index]["A"])

        tmplt = cv2.warpAffine(
            tmplt, A, dsize=(100, 100),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        )

        return {
            "idx": index,
            "template_affine": torch.Tensor(self.data[index]["template_affine"]).float().permute(-1, -2),
            "test_pts": torch.Tensor(self.data[index]['test_pts']).float().permute(-1, -2),
            "p_init": torch.Tensor(init),
            "img": self.transform(img),
            "tmplt": self.transform(tmplt)
        }


def collate_batch(batch):
    result = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([b[key] for b in batch], dim=0)
        else:
            result[key] = [b[key] for b in batch]
    return result


class FirstDataset(torch.utils.data.Dataset):
    def __init__(self, path: str):
        self.data_path = path
        self.file = h5py.File(self.data_path, "r")
        self.length = self.file["img"].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return {
            "idx": i,
            # 'M': torch.from_numpy(self.file['M'][i]).float(),
            'img': torch.from_numpy(self.file['img'][i]).float(),
            'tmplt': torch.from_numpy(self.file['tmplt'][i]).float(),
            # 'm': torch.from_numpy(self.file['m'][i]).float(),
            'p_init': torch.from_numpy(self.file['p_init'][i]).float(),
            # 'rms_pt_error': torch.tensor(self.file['rms_pt_error'][i]).float(),
            # 'rms_pt_init': torch.tensor(self.file['rms_pt_init'][i]).float(),
            'template_affine': torch.from_numpy(self.file['template_affine'][i]).float(),
            'test_pts': torch.from_numpy(self.file['test_pts'][i]).float(),
        }
