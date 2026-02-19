import torch
import time
import json
import os
import numpy as np
from torch.utils.data import DataLoader, random_split
from CPEN import CPEN
from Dataset import FirstDataset, collate_batch
from train_one_epoch import train_one_epoch
from evaluate_one_epoch import evaluate_one_epoch

if __name__ == '__main__':
    dt = torch.float32
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_epochs = 20
    batch = 2
    save_path = os.path.join(".", "pretrained_models")
    h5_path = '/home/ltopalis/Desktop/image-alignment-using-nn/dataset_matlab.hdf5'
    best_rms = float('inf')
    min_delta = 1e-4

    os.makedirs(save_path, exist_ok=True)

    full_ds = FirstDataset(h5_path)
    N = len(full_ds)
    n_train = int(0.8 * N)
    n_test = N - n_train

    generator = torch.Generator(device='cpu').manual_seed(42)

    train_ds, temp_ds = random_split(
        full_ds, [n_train, n_test], generator=generator)
    dev_ds, test_ds = random_split(
        temp_ds, [n_test // 2, n_test - n_test // 2], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=4, pin_memory=torch.cuda.is_available(),
                              persistent_workers=True, prefetch_factor=2,
                              collate_fn=collate_batch)
    dev_loader = DataLoader(dev_ds, batch_size=batch, shuffle=False,
                            num_workers=2, pin_memory=torch.cuda.is_available(),
                            persistent_workers=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False,
                             num_workers=2, pin_memory=torch.cuda.is_available(),
                             persistent_workers=True, collate_fn=collate_batch)

    print(
        f"Total samples: {N}, Train: {n_train}, Test: {n_test // 2}, Dev: {n_test - n_test // 2}")

    model = CPEN(levels=4, out_channels=8, device=dev, dtype=dt, DEBUG=True)
    model = model.to(device=dev, dtype=dt)

    # optimizer = torch.optim.Adam([
    #     {"params": model.model.parameters(), "lr": 1e-4},
    #     {"params": model.aggregator.parameters(), "lr": 0},
    # ])
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     factor=0.5,
    #     patience=2,
    #     threshold=1e-4,
    #     min_lr=1e-6,
    # )

    opt_cnn = torch.optim.Adam(model.model.parameters(), lr=1e-4)
    opt_agg = torch.optim.Adam(model.aggregator.parameters(), lr=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_cnn,
        mode='min',
        factor=0.5,
        patience=2,
        threshold=1e-4,
        min_lr=1e-6,
    )

    train_stats = {}
    start_time = time.perf_counter()
    for epoch in range(1, num_epochs + 1):
        print(f'======== epoch: {(epoch):2} ========')

        if epoch <= 4:
            for pg in opt_agg.param_groups:
                pg["lr"] = 0.0
            model.aggregator.temperature = 1.0
        elif epoch <= 6:
            for pg in opt_agg.param_groups:
                pg["lr"] = 1e-6
            model.aggregator.temperature = min(1.0, 0.5 + (epoch - 5) * 0.2)
        else:
            for pg in opt_agg.param_groups:
                pg["lr"] = 1e-5
            model.aggregator.temperature = min(1.0, 0.7 + (epoch - 6) * 0.1)

        epoch_sum_loss = 0.0
        epoch_num_samples = 0

        train_loss, max_rms, min_rms = train_one_epoch(
            model, train_loader, opt_cnn, opt_agg, dev, dt)
        idxs, RMSs = evaluate_one_epoch(model, dev_loader, dev, dt)

        if scheduler is not None:
            scheduler.step(float(np.mean(RMSs)))

        print(f"  Train loss : {train_loss:.4f}")
        print(f"    Max loss : {max_rms:.4f}")
        print(f"    Min loss : {min_rms:.4f}")
        print(f"  Dev loss   : {np.mean(RMSs):.4f}")

        train_stats[epoch] = {
            "train_loss": train_loss,
            "max_train_loss": max_rms,
            "min_train_loss": min_rms,
            "dev": {
                "rms": RMSs,
                "idx": idxs
            }
        }

        improved = np.mean(RMSs) < best_rms - min_delta

        if improved:
            best_rms = np.mean(RMSs)

            torch.save(model.state_dict(), os.path.join(
                save_path, "best_model.pth"))
            print(f"  New best model saved with RMS: {best_rms:.6f}")
        else:
            print(f"  No improvement over best RMS: {best_rms:.6f}")

        with open(os.path.join(save_path, "train_results.json"), "w") as f:
            json.dump(train_stats, f, indent=4)

    total_time = time.perf_counter() - start_time
    print(f'Total time: {total_time:.2f} s')

    print('======== TRAIN FINISH ========')

    model_fname = os.path.join(save_path, "best_model.pth")
    state = torch.load(model_fname, map_location=dev)
    model.load_state_dict(state)

    idxs, RMSs = evaluate_one_epoch(model, test_loader, dev, dt)

    print(f"Test RMS     : {np.mean(RMSs):.4f}")
    print(f"Test RMS std : {np.std(RMSs):.4f}")
    print(f"Test RMS min : {np.min(RMSs):.4f}")
    print(f"Test RMS max : {np.max(RMSs):.4f}")

    with open(os.path.join(save_path, "test_results.json"), "w") as f:
        json.dump({
            "test_rms_mean": np.mean(RMSs),
            "test_rms_std": np.std(RMSs),
            "test_rms_min": np.min(RMSs),
            "test_rms_max": np.max(RMSs),
            "test_idxs": idxs,
            "test_rms_values": RMSs
        }, f, indent=4)
