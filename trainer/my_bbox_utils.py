import numpy as np
import torch

def resize_boxes(target):
    """
    resize the bounding boxes according to the image resizing
    """
    for i in range(len(target['boxes'])):
        target['boxes'][i][0] = (512/target['dims'][i][0]) * target['boxes'][i][0]
        target['boxes'][i][1] = (512/target['dims'][i][1]) * target['boxes'][i][1]
        target['boxes'][i][2] = (512/target['dims'][i][0]) * target['boxes'][i][2]
        target['boxes'][i][3] = (512/target['dims'][i][1]) * target['boxes'][i][3]
    return target

def horizontal_flip(image, target):
    # img = torch.rot90(image, 2, [0, 1])
    img = image[:, ::-1, :]
    img_center = np.array(image.shape[:2])[::-1]/2
    img_center = np.hstack((img_center, img_center))
    target['boxes'][:, [0, 2]] = 2*(img_center[[0, 2]] - target['boxes'][:, [0, 2]])
    box_w = abs(target['boxes'][:, 0] - target['boxes'][:, 2])
    target['boxes'][:, 0] -= box_w
    target['boxes'][:, 2] += box_w
    return img, target

def normalize_boxes(target):
    """
    Divide x_min, x_max by width.
    Divide y_min, y_max by heigth.
    """
    for i in range(len(target['boxes'])):
        norm_1 = target['boxes'][i][0]/target['dims'][i][0]
        norm_2 = target['boxes'][i][1]/target['dims'][i][1]
        norm_3 = target['boxes'][i][2]/target['dims'][i][0]
        norm_4 = target['boxes'][i][3]/target['dims'][i][1]
        target['boxes'][i][0] = norm_1
        target['boxes'][i][1] = norm_2
        target['boxes'][i][2] = norm_3
        target['boxes'][i][3] = norm_4
    return target