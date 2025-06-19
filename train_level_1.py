import os
import torch
import argparse

from net import Level_1
from read_data import *
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Number of GPUs: {torch.cuda.device_count()}")

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_name", action="store",
                    dest="dataset_name", default="MSCOCO", help="MSCOCO")
parser.add_argument("--batch_size", action="store",
                    dest="batch_size", type=int, default=8, help="batch_size")
parser.add_argument("--output_channels", action="store", type=int,
                    dest="output_channels", default=8, help="8 -> homography, 4 -> affine")
parser.add_argument("--num_epochs", action="store", dest="num_epochs",
                    type=int, default=1, help="how many epochs to train")

input_args = parser.parse_args()

save_folder = os.path.join(
    "./checkpoints", input_args.dataset_name, "level_1")

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

network_level_1 = Level_1(
    out_channels=input_args.output_channels, device=device)

if input_args.dataset_name == "MSCOCO":
    dataset = dataSet_COCO("train")

dataloader = DataLoader(
    dataset, batch_size=input_args.batch_size, shuffle=True)

for epoch in range(input_args.num_epochs):
    for inp, x, y in dataloader:
        a = network_level_1(inp)
        print(a.size())
        break

    break
