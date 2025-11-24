import os
import torch
import argparse

from read_data import *
from initialMotion import HomographyRegressionHead
from featureExtractor import Level_1
from torch.utils.data import DataLoader
from pixel_ECC import pixel_ecc

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Number of GPUs: {torch.cuda.device_count()}")

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_name", action="store",
                    dest="dataset_name", default="MSCOCO", help="MSCOCO")
parser.add_argument("--batch_size", action="store",
                    dest="batch_size", type=int, default=8, help="batch_size")
parser.add_argument("--output_channels", action="store", type=int,
                    dest="output_channels", default=8, help="number of features")
parser.add_argument("--num_epochs", action="store", dest="num_epochs",
                    type=int, default=1, help="how many epochs to train")
parser.add_argument("--epoch_start", action="store", dest="epoch_start",
                    type=int, default=20, help="train from which epoch")
parser.add_argument("--learning_rate", action="store", dest="lr",
                    type=float, default=0.001, help="learning rate for Adam optimizer")
<<<<<<< Updated upstream
=======
parser.add_argument("--steps", action="store", dest="steps",
                    type=int, default=200, help="number of iterations")
parser.add_argument("--loss_type", action="store", dest="loss",
                    default="l2", help="loss function - l1, l2, linf")
>>>>>>> Stashed changes

input_args = parser.parse_args()

save_folder = os.path.join(
    "./checkpoints", input_args.dataset_name, "level_1")

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

network_level_1 = Level_1(
    out_channels=input_args.output_channels, device=device)

if input_args.epoch_start > 1:
    load_path = os.path.join(
        save_folder, f"epoch_{input_args.epoch_start - 1}_model.pth")

    if not os.path.exists(load_path):
        raise FileExistsError(
            f"Epoch {input_args.epoch_start - 1} doesn't exist!")

    model_weights = torch.load(load_path, weights_only=True)
    network_level_1.load_state_dict(model_weights)

if input_args.dataset_name == "MSCOCO":
    dataset = dataSet_COCO("train")

dataloader = DataLoader(
    dataset, batch_size=input_args.batch_size, shuffle=True)

for epoch in range(input_args.num_epochs - input_args.epoch_start + 1):

    for inp, template, x, y in dataloader:
        F_I = network_level_1(inp)
        F_T = network_level_1(template)

<<<<<<< Updated upstream
=======
        initial_motion = torch.Tensor(
            [[1, 0, 0, 0, 1, 0, 0, 0]for _ in range(input_args.batch_size)]).float()

        pixel_ecc(F_T, F_I, initial_motion, input_args.lr,
                  input_args.steps, input_args.loss)

>>>>>>> Stashed changes
        break
    break
    save_path = os.path.join(
        save_folder, f"epoch_{input_args.epoch_start + epoch}_model.pth")
    torch.save(network_level_1.state_dict(), save_path)
