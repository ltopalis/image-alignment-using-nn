import torch
import json
import time
import csv
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from CPEN import CPEN
from Dataset import H5Dataset, collate_batch
from pixel_ecc_affine.ComputePointError import ComputePointError

torch.set_default_device('cpu')

if __name__ == '__main__':
    # dt = torch.float64
    dt = torch.float32
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_epochs = 20
    batch = 3
    save_path = os.path.join(".", "pretrained_models")
    h5_path = 'data/dataset/data.h5'

    os.makedirs(save_path, exist_ok=True)

    full_ds = H5Dataset(h5_path, dtype=dt)
    N = len(full_ds)
    n_train = int(0.8 * N)
    n_test = N - n_train

    train_ds, test_ds = random_split(full_ds, [n_train, n_test],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=4, pin_memory=torch.cuda.is_available(),
                              persistent_workers=True, prefetch_factor=2,
                              collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False,
                             num_workers=2, pin_memory=torch.cuda.is_available(),
                             persistent_workers=True, collate_fn=collate_batch)

    # data_mat = loadmat(
    #     '/Users/ltopalis/Library/Mobile Documents/com~apple~CloudDocs/Documents/THESIS/IMPLEMENTATION/5 - pixel-ECC/to_dimitri_run_ecc_pixel/a.mat'
    # )['data'][0][0]

    # data = []
    # n_samples = int(data_mat['iter_'][0][0])
    # for i in range(n_samples):
    #     data.append({
    #         'idx': i,
    #         'img': torch.from_numpy(data_mat['img'][i, :, :]).to(dtype=dt, device=dev),
    #         'tmplt': torch.from_numpy(data_mat['tmplt'][i, :, :]).to(dtype=dt, device=dev),
    #         'M_gt': torch.from_numpy(data_mat['M'][i, :, :]).to(dtype=dt, device=dev),
    #         'test_pts': torch.from_numpy(data_mat['test_pts'][i, :, :]).to(dtype=dt, device=dev),
    #         'template_affine': torch.from_numpy(data_mat['template_affine'][i, :, :]).to(dtype=dt, device=dev)
    #     })

    # random.shuffle(data)
    # split_idx = math.floor(len(data) * 0.8)
    # if split_idx % 2 == 1:
    #     split_idx += 1

    # train_data = data[:split_idx]
    # test_data = data[split_idx:]

    print(
        f"Total samples: {N}, Train: {n_train}, Test: {n_test}")

    model = CPEN(levels=3, out_channels=32, device=dev, dtype=dt)
    model = model.to(device=dev, dtype=dt)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    results = {}
    avg_loss = []
    str_train_results = "epoch        min_loss     max_loss     average_loss    elapsed_time_s\n"

    start_time = time.perf_counter()
    for epoch in range(num_epochs):
        print(f'======== epoch: {(epoch + 1):2} ========')
        results[epoch] = {'rms': [], 'idxs': []}

        # random.shuffle(train_data)

        model.train()
        epoch_sum_loss = 0.0
        epoch_num_samples = 0

        # n_pairs = len(train_data) // 2
        # batch_iter = range(0, n_pairs * 2, 2)

        # for i in tqdm(batch_iter, desc=f"Epoch {epoch+1} batches"):
        for data in tqdm(train_loader, desc=f'Epoch {epoch+1} batches'):
            img = data['img'].unsqueeze(1).to(dtype=dt, device=dev)
            tmplt = data['tmplt'].unsqueeze(1).to(dtype=dt, device=dev)
            test_pts = data['test_pts'].to(dtype=dt, device=dev)
            template_affine = data['template_affine'].to(dtype=dt, device=dev)
            # img1 = train_data[i]['img'].unsqueeze(0).unsqueeze(0)
            # tmplt1 = train_data[i]['tmplt'].unsqueeze(0).unsqueeze(0)
            # test_pts1 = train_data[i]['test_pts'].unsqueeze(0).unsqueeze(0)
            # template_affine1 = train_data[i]['template_affine'].unsqueeze(
            #     0).unsqueeze(0)

            # img2 = train_data[i + 1]['img'].unsqueeze(0).unsqueeze(0)
            # tmplt2 = train_data[i + 1]['tmplt'].unsqueeze(0).unsqueeze(0)
            # test_pts2 = train_data[i + 1]['test_pts'].unsqueeze(0).unsqueeze(0)
            # template_affine2 = train_data[i +
            #                               1]['template_affine'].unsqueeze(0).unsqueeze(0)

            # img = torch.cat([img1, img2], dim=0)         # (2,1,H,W)
            # tmplt = torch.cat([tmplt1, tmplt2], dim=0)  # (2,1,H,W)
            # test_pts = torch.cat([test_pts1, test_pts2], dim=0).squeeze(
            #     1)            # (2, N, 2)
            # template_affine = torch.cat(
            #     [template_affine1, template_affine2], dim=0).squeeze(1)

            pred = model(img, tmplt)
            m = torch.zeros((pred.shape[0], 2, 3), dtype=dt, device=dev)
            rms = ComputePointError(test_pts, template_affine, pred, m)

            loss = rms.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_sample_count = img.shape[0]
            epoch_sum_loss += loss.item() * batch_sample_count
            epoch_num_samples += batch_sample_count

            results[epoch]["idxs"].extend([int(x) for x in data['idx']])
            rms_flat = torch.flatten(rms).detach().cpu().tolist()
            results[epoch]["rms"].extend([float(x) for x in rms_flat])

        if epoch_num_samples > 0:
            epoch_avg_loss = epoch_sum_loss / epoch_num_samples
        else:
            epoch_avg_loss = float('nan')

        avg_loss.append(epoch_avg_loss)
        epoch_min = min(results[epoch]['rms']
                        ) if results[epoch]['rms'] else float('nan')
        epoch_max = max(results[epoch]['rms']
                        ) if results[epoch]['rms'] else float('nan')

        elapsed = time.perf_counter() - start_time
        str_train_results += f"{epoch:5d}{epoch_min:14.6e}{epoch_max:14.6e}{epoch_avg_loss:14.6e}{elapsed:12.2f}\n"

        print(f'\taverage loss : {epoch_avg_loss:.6e}')
        print(f'\tmax loss      : {epoch_max:.6e}')
        print(f'\tmin loss      : {epoch_min:.6e}')

        model_fname = os.path.join(save_path, f"model_epoch_{epoch:02d}.pth")
        torch.save(model.state_dict(), model_fname)
        print(f"Saved model weights to {model_fname}")

    print('======== TRAIN FINISH ========')

    if avg_loss:
        mean_of_avgs = sum(avg_loss) / len(avg_loss)
        best_epoch = int(avg_loss.index(min(avg_loss)))
        worst_epoch = int(avg_loss.index(max(avg_loss)))
        print(f'Average epoch loss: {mean_of_avgs:.6e}')
        print(
            f'Best model at epoch: {best_epoch} with value {min(avg_loss):.6e}')
        print(
            f'Worst model at epoch: {worst_epoch} with value {max(avg_loss):.6e}')
    else:
        print("No epochs were run.")

    total_time = time.perf_counter() - start_time
    print(f'Total time: {total_time:.2f} s')

    with open(os.path.join(save_path, "pretrained_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(save_path, 'pretrained_results.txt'), 'w') as f:
        f.write(str_train_results)

    best_model_idx = int(avg_loss.index(min(avg_loss))) if avg_loss else None

    model_fname = os.path.join(
        save_path, f"model_epoch_{best_model_idx:02d}.pth")
    state = torch.load(model_fname, map_location=dev)
    model.load_state_dict(state)

    all_errors = []          # per-sample scalar error (mean RMS per sample)
    all_errors_perpoint = []
    indices = []             # sample indices (keep track)
    pred_matrices = []       # predicted matrices if available
    # ground truth matrices if available (template_affine or M_gt)
    gt_matrices = []

    model.eval()
    with torch.no_grad():
        #     n_pairs = len(test_data) // 2
        #     batch_iter = range(0, n_pairs * 2, 2)

        for data in tqdm(test_loader, desc=f"Epoch {epoch+1} batches"):
            img = data['img'].unsqueeze(1).to(dtype=dt, device=dev)
            tmplt = data['tmplt'].unsqueeze(1).to(dtype=dt, device=dev)
            test_pts = data['test_pts'].to(dtype=dt, device=dev)
            template_affine = data['template_affine'].to(dtype=dt, device=dev)
    #         img1 = test_data[i]['img'].unsqueeze(0).unsqueeze(0)
    #         tmplt1 = test_data[i]['tmplt'].unsqueeze(0).unsqueeze(0)
    #         test_pts1 = test_data[i]['test_pts'].unsqueeze(0).unsqueeze(0)
    #         template_affine1 = test_data[i]['template_affine'].unsqueeze(
    #             0).unsqueeze(0)

    #         img2 = test_data[i + 1]['img'].unsqueeze(0).unsqueeze(0)
    #         tmplt2 = test_data[i + 1]['tmplt'].unsqueeze(0).unsqueeze(0)
    #         test_pts2 = test_data[i + 1]['test_pts'].unsqueeze(0).unsqueeze(0)
    #         template_affine2 = test_data[i +
    #                                      1]['template_affine'].unsqueeze(0).unsqueeze(0)

    #         img = torch.cat([img1, img2], dim=0)         # (2,1,H,W)
    #         tmplt = torch.cat([tmplt1, tmplt2], dim=0)  # (2,1,H,W)
    #         test_pts = torch.cat([test_pts1, test_pts2], dim=0).squeeze(
    #             1)            # (2, N, 2)
    #         template_affine = torch.cat(
    #             [template_affine1, template_affine2], dim=0).squeeze(1)

            pred = model(img, tmplt)
            rms = ComputePointError(test_pts, template_affine, pred, m)

            if isinstance(rms, torch.Tensor):
                rms_arr = rms.detach().cpu().numpy().reshape(-1)   # shape (B,)
            else:
                rms_arr = np.array(rms).reshape(-1)

            if rms_arr.ndim == 2:
                per_sample_mean = np.mean(rms_arr, axis=1)   # (B,)
                for r in rms_arr:
                    all_errors_perpoint.append(np.asarray(r))
            else:
                per_sample_mean = rms_arr

            for bi in range(per_sample_mean.shape[0]):
                idx = data['idx'][bi]
                indices.append(int(idx))
                all_errors.append(float(per_sample_mean[bi]))
                p = pred[bi].detach().cpu().numpy()
                if p.shape == (2, 3) or p.shape == (3, 3):
                    pred_matrices.append(p.copy())
                    gt_matrices.append(
                        template_affine[bi].detach().cpu().numpy())

    errors = np.array(all_errors)
    stats = {
        'n_samples': int(errors.size),
        'mean': float(np.mean(errors)),
        'median': float(np.median(errors)),
        'std': float(np.std(errors)),
        'min': float(np.min(errors)),
        'max': float(np.max(errors)),
    }

    with open(os.path.join(save_path, 'evaluation_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    csv_path = os.path.join(save_path, 'per_sample_errors.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['idx', 'error_px'])
        for idx, err in zip(indices, all_errors):
            writer.writerow([idx, err])

    print("Evaluation stats summary:")
    for k, v in stats.items():
        print(f"{k:25s}: {v}")

    with open(os.path.join(save_path, 'evaluation_summary.txt'), 'w') as f:
        f.write("Evaluation summary\n")
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
