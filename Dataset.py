import h5py
from torch.utils.data import Dataset
import os
import torch
import numpy as np
from scipy.io import loadmat
from pixel_ecc_affine.spatial_interp import spatial_interp


class Dataset(torch.utils.data.Dataset):
    def __init__(self, mat_path: str,
                 spatial_sigma: list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                 num_samples: int = 100_000,
                 device: torch.device | str = 'cpu',
                 dtype: torch.dtype = torch.float32):

        data = loadmat(mat_path)

        self.num_of_subjs = data['num_of_subjs'].item()
        self.example_imgs = data['example_imgs']
        self.tmplts = data['tmplts']
        # self.roi = data['coords'][0]  # [x1 x2 y1 y2]
        self.roi = [50, 80, 177, 207]
        self.pt_offset = data['pt_offset']

        self.sigmas = spatial_sigma
        self.num_samples = num_samples
        self.dev = device
        self.dt = dtype

    def __getitem__(self, idx):
        tmplt_idx = np.random.randint(self.num_of_subjs)
        wimage_idx = tmplt_idx * 10 + np.random.randint(10)

        tmplt = self.tmplts[:, :, tmplt_idx]
        wimage = self.example_imgs[:, :, wimage_idx]

        sigma = np.random.choice(self.sigmas)

        target_pts = torch.tensor([[self.roi[0], self.roi[0], self.roi[2], self.roi[2]],
                                   [self.roi[1], self.roi[3], self.roi[1], self.roi[3]]], device=self.dev, dtype=self.dt)

        target_affine = torch.tensor([[self.roi[0], self.roi[2], self.roi[0] + ((self.roi[2] - self.roi[0]) / 2) - 0.5],
                                      [self.roi[1], self.roi[1], self.roi[3]]], device=self.dev, dtype=self.dt)

        template_nx = self.roi[2] - self.roi[0] + 1
        template_ny = self.roi[3] - self.roi[1] + 1

        template_pts = torch.tensor([[1, 1, template_nx, template_nx], [
                                    1, template_ny, template_ny, 1]], device=self.dev, dtype=self.dt)

        template_affine = torch.tensor(
            [[1, template_nx, template_nx / 2],
             [1, 1, template_ny]], device=self.dev, dtype=self.dt)

        ind = np.random.randint(
            0, self.pt_offset.shape[0], size=1, dtype=np.int32)

        pt_offsets1 = self.pt_offset[ind, :]
        pt_offsets1 = pt_offsets1 * sigma

        pt_offsets1 = torch.from_numpy(pt_offsets1)
        pt_offsets1 = pt_offsets1.to(device=self.dev, dtype=self.dt)

        test_pts = target_affine + pt_offsets1.reshape(2, 3)
        test_pts = test_pts.to(device=self.dev, dtype=self.dt)

        A = torch.cat([template_affine.T, torch.ones(
            3, 1, device=self.dev, dtype=self.dt)], dim=1)
        B = test_pts.T
        M = torch.linalg.lstsq(A, B).solution.T

        ny = torch.arange(template_ny, dtype=self.dt, device=self.dev)
        nx = torch.arange(template_nx, dtype=self.dt, device=self.dev)

        tmplt = torch.from_numpy(tmplt)
        tmplt = tmplt.to(device=self.dev, dtype=self.dt)
        tmplt = tmplt.unsqueeze(0)
        tmplt = torch.nn.functional.interpolate(tmplt, size=256, mode='linear')
        tmplt = tmplt[:, 21:-21, :]

        wimage = torch.from_numpy(wimage)
        wimage = wimage.to(device=self.dev, dtype=self.dt)
        wimage = wimage.unsqueeze(0)
        wimage = torch.nn.functional.interpolate(
            wimage, size=256, mode='linear')
        wimage = wimage[:, 21:-21, :]
        wimage = spatial_interp(wimage.unsqueeze(
            0), M, 'linear', 'affine', nx, ny).squeeze(0)

        return_values = {
            'template': tmplt,
            'wimage': wimage,
            'test_pts': test_pts,
            'template_affine': template_affine,
            'affine_gt': M
        }

        return return_values

    def __len__(self):
        return self.num_samples


class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, keys=('img', 'tmplt', 'M', 'test_pts', 'template_affine'),
                 dtype=torch.float32):
        self.h5_path = h5_path
        self._file = None
        self._len = None
        self.keys = list(keys)
        self.dtype = dtype

    def _ensure_open(self):
        if self._file is None:
            import h5py
            self._file = h5py.File(self.h5_path, 'r')

    def __len__(self):
        if self._len is None:
            import h5py
            with h5py.File(self.h5_path, 'r') as f:
                self._len = f[self.keys[0]].shape[0]
        return int(self._len)

    def __getitem__(self, idx):
        self._ensure_open()
        if idx < 0:
            idx = len(self) + idx

        sample = {}
        for k in self.keys:
            arr = self._file[k][idx]
            t = torch.as_tensor(arr, dtype=self.dtype)  # πάντα CPU
            sample[k] = t
        sample['idx'] = int(idx)
        return sample

    def close(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    def __del__(self):
        self.close()


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
        self.length = self.file["M"].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return {
            "idx": i,
            'M': torch.from_numpy(self.file['M'][i]).double(),
            'img': torch.from_numpy(self.file['img'][i]).double(),
            'tmplt': torch.from_numpy(self.file['tmplt'][i]).double(),
            'm': torch.from_numpy(self.file['m'][i]).double(),
            'p_init': torch.from_numpy(self.file['p_init'][i]).double(),
            'rms_pt_error': torch.tensor(self.file['rms_pt_error'][i]).double(),
            'rms_pt_init': torch.tensor(self.file['rms_pt_init'][i]).double(),
            'template_affine': torch.from_numpy(self.file['template_affine'][i]).double(),
            'test_pts': torch.from_numpy(self.file['test_pts'][i]).double(),
        }


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    d = FirstDataset(
        "/home/ltopalis/Desktop/image-alignment-using-nn/dataset_matlab.hdf5")

    # d = Dataset(
    #     '/home/ltopalis/Desktop/image-alignment-using-nn/dataset_matlab.hdf5')

    j = 1
    for i in range(10):
        a = d[i]

        wimage = a['img'].squeeze().detach().cpu()
        template = a['tmplt'].squeeze().detach().cpu()

        plt.subplot(2, 5, (j % 5) + 1)
        plt.imshow(template, cmap=plt.cm.gray)
        plt.axis('off')
        plt.tight_layout()

        plt.subplot(2, 5, 5 + (j % 5) + 1)
        plt.imshow(wimage, cmap=plt.cm.gray)
        plt.axis('off')
        plt.tight_layout()

        j += 1

        print(template.shape, wimage.shape)
    plt.savefig('plot3.png', dpi=200)
