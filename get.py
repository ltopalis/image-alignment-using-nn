import h5py
import torch
import numpy as np
from pixel_ecc_affine.ECC_PIXEL_IA import ECC_PIXEL_IA
from pixel_ecc_affine.ComputePointError import ComputePointError

dt = torch.float64
dev = 'cpu'

with h5py.File('/home/ltopalis/Desktop/paparia/data/dataset/results.h5', 'r') as file:
    print(file['default'])

# with h5py.File('/home/ltopalis/Desktop/paparia/data/dataset/data.h5', 'r') as file:
#     M = torch.as_tensor(np.array(file['M']), dtype=dt, device=dev)
#     img = torch.as_tensor(np.array(file['img']), dtype=dt, device=dev)
#     tmplt = torch.as_tensor(np.array(file['tmplt']), dtype=dt, device=dev)
#     template_affine = torch.as_tensor(
#         np.array(file['template_affine']), device=dev, dtype=dt)
#     test_pts = torch.as_tensor(
#         np.array(file['test_pts']), device=dev, dtype=dt)

# num_samples = img.shape[0]

# res = torch.zeros(num_samples, device=dev, dtype=dt)

# for i in range(num_samples):
#     fitt = ECC_PIXEL_IA(img[i, :, :].unsqueeze(0).unsqueeze(0), tmplt[i, :, :].unsqueeze(0).unsqueeze(0), torch.tensor(
#         [[0, 0, 49], [0, 0, 79]], device=dev, dtype=dt).unsqueeze(0).unsqueeze(0))
#     rms = ComputePointError(test_pts[i, :, :].unsqueeze(0), template_affine[i, :, :].unsqueeze(
#         0), fitt[-1]['warp_p'], torch.tensor([[0, 0, 0], [0, 0, 0]], device=dev, dtype=dt).unsqueeze(0))
#     res[i] = rms.squeeze()

#     if i % 100 == 0:
#         print(i)
#         with h5py.File('/home/ltopalis/Desktop/paparia/data/dataset/results.h5', 'w') as f:
#             dset = f.create_dataset("default", data=res)


# with h5py.File('/home/ltopalis/Desktop/paparia/data/dataset/results.h5', 'w') as f:
#     dset = f.create_dataset("default", data=res)
