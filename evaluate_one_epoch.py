import torch
from tqdm import tqdm
from pixel_ecc_affine.ComputePointError import ComputePointError


@torch.no_grad()
def evaluate_one_epoch(model, dataloader, device='cpu', dtype=torch.float32):
    model.eval()
    all_rms = []
    all_idxs = []

    for data in tqdm(dataloader, desc='Evaluating', unit='batch'):

        img = data['img'].unsqueeze(1).to(
            dtype=dtype, device=device) if data['img'].ndim != 4 else data['img'].to(dtype=dtype, device=device)
        tmplt = data['tmplt'].unsqueeze(1).to(
            dtype=dtype, device=device) if data['tmplt'].ndim != 4 else data['tmplt'].to(dtype=dtype, device=device)
        test_pts = data['test_pts'].to(dtype=dtype, device=device)
        template_affine = data['template_affine'].to(
            dtype=dtype, device=device)
        p_init = data['p_init'].to(dtype=dtype, device=device)

        pred = model(img, tmplt, init_p=p_init)
        m = torch.zeros((pred.shape[0], 2, 3), dtype=dtype, device=device)
        rms = ComputePointError(test_pts, template_affine, pred, m)

        all_idxs.extend([int(x) for x in data['idx']])
        rms_flat = torch.flatten(rms).detach().cpu().tolist()
        all_rms.extend([float(x) for x in rms_flat])

    return all_idxs, all_rms
