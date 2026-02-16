import torch
from tqdm import tqdm
from pixel_ecc_affine.ComputePointError import ComputePointError


def train_one_epoch(model, train_loader, optimizer_cnn, optimizer_agg, device, dtype):
    model.train()
    epoch_sum_loss = 0.0
    epoch_num_samples = 0
    max_epoch = -float('inf')
    min_epoch = float('inf')
    skipped = 0

    for data in tqdm(train_loader, desc=f'Training', unit='batch'):
        img = data['img'].unsqueeze(1).to(
            dtype=dtype, device=device, non_blocking=True)
        tmplt = data['tmplt'].unsqueeze(1).to(
            dtype=dtype, device=device, non_blocking=True)
        test_pts = data['test_pts'].to(dtype=dtype, device=device)
        template_affine = data['template_affine'].to(
            dtype=dtype, device=device)
        p_init = data['p_init'].to(
            dtype=dtype, device=device, non_blocking=True)

        optimizer_cnn.zero_grad(set_to_none=True)
        optimizer_agg.zero_grad(set_to_none=True)
        pred = model(img, tmplt, init_p=p_init)
        if not torch.isfinite(pred).all():
            skipped += 1
            continue
        m = torch.zeros((pred.shape[0], 2, 3), dtype=dtype, device=device)
        rms = ComputePointError(test_pts, template_affine, pred, m)
        if not torch.isfinite(rms).all():
            continue

        max_batch_rms = rms.max().item()
        if max_batch_rms > max_epoch:
            max_epoch = max_batch_rms
        min_batch_rms = rms.min().item()
        if min_batch_rms < min_epoch:
            min_epoch = min_batch_rms
        loss = rms.mean()

        loss.backward()

        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0)
        if not torch.isfinite(total_norm):
            optimizer_cnn.zero_grad(set_to_none=True)
            optimizer_agg.zero_grad(set_to_none=True)
            continue

        optimizer_cnn.step()
        optimizer_agg.step()

        batch_sample_count = img.shape[0]
        epoch_sum_loss += loss.item() * batch_sample_count
        epoch_num_samples += batch_sample_count

    if epoch_num_samples > 0:
        epoch_avg_loss = epoch_sum_loss / epoch_num_samples
    else:
        epoch_avg_loss = float('nan')

    print(f"Skipped batches this epoch: {skipped * 100 / len(train_loader)}%")

    return epoch_avg_loss, max_epoch, min_epoch
