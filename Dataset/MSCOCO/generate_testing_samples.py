import os
import cv2
import json
import glob
import random

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from skimage.transform import resize

root_folder = '/Users/ltopalis/Library/CloudStorage/GoogleDrive-lazarostop32@gmail.com/Το Drive μου/THESIS/Dataset/MS-COCO/images/'
all_images_path = "/Users/ltopalis/Library/CloudStorage/GoogleDrive-lazarostop32@gmail.com/Το Drive μου/THESIS/Dataset/MS-COCO/zipFiles/val2017/*"
val2017_template = os.path.join(
    root_folder, 'val2017_template/')
val2017_label = os.path.join(
    root_folder, 'val2017_label/')
val2017_input = os.path.join(
    root_folder, 'val2017_input/')

img_path_list = glob.glob(all_images_path)

if not (os.path.exists(val2017_template)):
    os.makedirs(val2017_template)
if not (os.path.exists(val2017_label)):
    os.makedirs(val2017_label)
if not (os.path.exists(val2017_input)):
    os.makedirs(val2017_input)


index = 0
for img_path in img_path_list:
    index += 1
    print(index)

    if index > 6400:
        break

    img = plt.imread(img_path)
    img_name = img_path.split('/')[-1]

    new_img = resize(img, (240, 320))
    if len(np.shape(new_img)) < 3:
        print('!!!!!!!!')
        continue

    random_top_left_x = random.randint(0, 40)
    random_top_left_y = random.randint(0, 100)

    square_img = new_img[random_top_left_x:random_top_left_x +
                         192, random_top_left_y:random_top_left_y+192, :]
    plt.imsave(os.path.join(val2017_input, img_name), square_img)

    top_left_box_x = random.randint(0, 63)
    top_left_box_y = random.randint(0, 63)

    top_right_box_x = random.randint(128, 191)
    top_right_box_y = random.randint(0, 63)

    bottom_left_box_x = random.randint(0, 63)
    bottom_left_box_y = random.randint(128, 191)

    bottom_right_box_x = random.randint(128, 191)
    bottom_right_box_y = random.randint(128, 191)

    src_points = [[top_left_box_x, top_left_box_y],
                  [top_right_box_x, top_right_box_y],
                  [bottom_left_box_x, bottom_left_box_y],
                  [bottom_right_box_x, bottom_right_box_y]]

    tgt_points = [[32, 32],
                  [159, 32],
                  [32, 159],
                  [159, 159]]

    src_points = np.reshape(src_points, [4, 1, 2])
    tgt_points = np.reshape(tgt_points, [4, 1, 2])

    # find homography
    h_matrix, status = cv2.findHomography(src_points, tgt_points, 0)

    simulated_drone_img = cv2.warpPerspective(square_img, h_matrix, (192, 192))
    plt.imsave(os.path.join(val2017_template, img_name),
               simulated_drone_img[32:160, 32:160, :])

    label = {}
    label['location'] = []

    label['location'].append({
        'top_left_x': top_left_box_x,
        'top_left_y': top_left_box_y
    })
    label['location'].append({
        'top_right_x': top_right_box_x,
        'top_right_y': top_right_box_y
    })
    label['location'].append({
        'bottom_left_x': bottom_left_box_x,
        'bottom_left_y': bottom_left_box_y
    })
    label['location'].append({
        'bottom_right_x': bottom_right_box_x,
        'bottom_right_y': bottom_right_box_y
    })

    with open(os.path.join(val2017_label, f"{img_name[:(len(img_name)-4)]}_label.txt"), 'w') as outfile:
        json.dump(label, outfile)
