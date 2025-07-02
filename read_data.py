import os
import glob
import json
import torch

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class dataSet_COCO(Dataset):

    def __init__(self, dataset_name="train", target_size=(192, 192)):

        _root_folder = "/Users/ltopalis/Library/CloudStorage/GoogleDrive-lazarostop32@gmail.com/Το Drive μου/THESIS/Dataset/MS-COCO"
        self.target_size = target_size

        if dataset_name == "train":
            self.img_path = glob.glob(
                os.path.join(_root_folder, "images", "train2017_input", "*"))
            self.input_path = os.path.join(
                _root_folder, "images", "train2017_input")
            self.label_path = os.path.join(
                _root_folder, "images", "train2017_label")
            self.template_path = os.path.join(
                _root_folder, "images", "train2017_template")

        elif dataset_name == "val":
            self.img_path = glob.glob(
                os.path.join(_root_folder, "images", "val2017_input", "*"))
            self.input_path = os.path.join(
                _root_folder, "images", "val2017_input")
            self.label_path = os.path.join(
                _root_folder, "images", "val2017_label")
            self.template_path = os.path.join(
                _root_folder, "images", "val2017_template")

        else:
            raise FileNotFoundError

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        image_path = self.img_path[idx]
        img_name = image_path.split("/")[-1]

        input_img = plt.imread(os.path.join(self.input_path, img_name)) / 255.0
        template_img = plt.imread(os.path.join(
            self.template_path, img_name)) / 255.0

        if input_img.ndim == 3:
            input_img = TF.rgb_to_grayscale(
                torch.tensor(input_img).permute(2, 0, 1))
        else:
            input_img = torch.tensor(input_img).permute(2, 0, 1)

        if template_img.ndim == 3:
            template_img = TF.rgb_to_grayscale(
                torch.tensor(template_img).permute(2, 0, 1))
        else:
            template_img = torch.tensor(template_img).permute(2, 0, 1)

        input_img = TF.resize(input_img, self.target_size)
        template_img = TF.resize(template_img, self.target_size)

        with open(os.path.join(self.label_path, f"{img_name[:(len(img_name)-4)]}_label.txt"), "r") as f:
            data = json.load(f)

        x_list = [data['location'][0]['top_left_x'], data['location'][1]['top_right_x'],
                  data['location'][3]['bottom_right_x'], data['location'][2]['bottom_left_x']]
        y_list = [data['location'][0]['top_left_y'], data['location'][1]['top_right_y'],
                  data['location'][3]['bottom_right_y'], data['location'][2]['bottom_left_y']]

        return (np.asarray(input_img).astype(np.float32), np.asarray(template_img).astype(np.float32),
                np.asarray(x_list).astype(np.float32), np.asarray(y_list).astype(np.float32))
