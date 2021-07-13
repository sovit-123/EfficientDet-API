import pandas as pd
import torch
import random
import numpy as np
import cv2
from torch._C import dtype

from transforms import get_train_transforms, get_valid_transforms
from torch.utils.data import Dataset
from helpers import viz_train_augment
from my_bbox_utils import resize_boxes, normalize_boxes, horizontal_flip

ROOT_PATH = '..'

TRAIN_ROOT_PATH = f"{ROOT_PATH}/final_train_images"
VALID_ROOT_PATH = f"{ROOT_PATH}/final_valid_images"
TEST_ROOT_PATH = f"{ROOT_PATH}/final_test_images"

train_csv = pd.read_csv(f"{ROOT_PATH}/final_labels/final_train_csv.csv")
valid_csv = pd.read_csv(f"{ROOT_PATH}/final_labels/final_valid_csv.csv")
test_csv = pd.read_csv(f"{ROOT_PATH}/final_labels/final_test_csv.csv")

train_df = train_csv[['filename']].copy()
train_df.loc[:, 'bbox_count'] = 1
train_df = train_df.groupby('filename').count()

valid_df = valid_csv[['filename']].copy()
valid_df.loc[:, 'bbox_count'] = 1
valid_df = valid_df.groupby('filename').count()

test_df = test_csv[['filename']].copy()
test_df.loc[:, 'bbox_count'] = 1
test_df = test_df.groupby('filename').count()

class ODDataset(Dataset):

    def __init__(
        self, marking, image_ids, transforms=None, test=False, ROOT_PATH=None
    ):
        super().__init__()

        self.image_ids = image_ids
        self.marking = marking
        self.transforms = transforms
        self.test = test
        self.root_path = ROOT_PATH

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]

        ##### UNCOMMENT THE FOLLOWING LINE IF YOU DO NOT WANT TO USE CUTMIX #####
        image, boxes, classes, dims = self.load_image_and_boxes(index, self.root_path)

        # ##### THIS FOR CUTMIX AUGMENTATIONS (COMMENT OUT IF GETTING ERRORS) #####
        # if self.test or random.random() > 0.5:
        #     image, boxes, classes, dims = self.load_image_and_boxes(index, self.root_path)
        # else:
        #     image, boxes, classes, dims = self.load_cutmix_image_and_boxes(index, self.root_path)
        ##### THE ABOVE FOR CUTMIX AUGMENTATIONS #####

        CLASSES = ['boat']
        labels = []
        for i in range(len(classes)):
            label = CLASSES.index(classes[i]) + 1
            labels.append(label)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['dims'] = dims
        # print(abs(target['dims'][0][0] - target['dims'][0][1]))

        if self.transforms:
            for i in range(40):
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels,
                    'transformed': not False
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float)
                    target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  #yxyx: be warning
                    # print('-' * 30)
                    break

                # if mosaic augmentation did not return any bounding boxes, then again run augmentations
                # if len(sample['bboxes']) == 0: 
                #     image, boxes, classes, dims = self.load_cutmix_image_and_boxes(index, self.root_path)
                #     labels = []
                #     for i in range(len(classes)):
                #         label = CLASSES.index(classes[i]) + 1
                #         labels.append(label)

                #     labels = torch.tensor(labels, dtype=torch.int64)
                    
                #     target = {}
                #     target['boxes'] = boxes
                #     target['labels'] = labels
                #     target['image_id'] = torch.tensor([index])
                #     target['dims'] = dims
                #     continue
        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def load_image_and_boxes(self, index, root_path):
        # print(self.image_ids[index])
        image_id = self.image_ids[index]
        image = cv2.imread(f"{root_path}/{image_id}", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        records = self.marking[self.marking['filename'] == image_id]
        dims = records[['width', 'height']].values
        # dims = records[['height', 'width']].values
        boxes = records[['xmin', 'ymin', 'xmax', 'ymax']].values
        classes = records[['class']].values
        return image, boxes, classes, dims

    def load_cutmix_image_and_boxes(self, index, root_path):
        """ 
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia 
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        image_id = self.image_ids[index]
        image = cv2.imread(f"{root_path}/{image_id}", cv2.IMREAD_COLOR) # we need to read image each time as the height and width of each image might not be the same
        h, w, c = image.shape
        s = h // 2
    
        xc, yc = [int(random.uniform(h * 0.25, w * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]

        result_image = np.full((h, w, 3), 1, dtype=np.float32)
        result_boxes = []
        result_classes = []

        for i, index in enumerate(indexes):
            image, boxes, classes, dims = self.load_image_and_boxes(index, self.root_path)

            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)
            for class_name in classes:
                result_classes.append(class_name) 
            
        final_classes = []
        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)

        for idx in range(len(result_boxes)):
            if ((result_boxes[idx,2]-result_boxes[idx,0])*(result_boxes[idx,3]-result_boxes[idx,1])) > 0:
                final_classes.append(result_classes[idx])

        result_boxes = result_boxes[np.where((result_boxes[:,2]-result_boxes[:,0])*(result_boxes[:,3]-result_boxes[:,1]) > 0)]
        return result_image, result_boxes, final_classes, dims

train_dataset = ODDataset(
    image_ids=train_df.index.values, marking=train_csv, 
    transforms=get_train_transforms(), test=False, ROOT_PATH=TRAIN_ROOT_PATH
)

valid_dataset = ODDataset(
    image_ids=valid_df.index.values, marking=valid_csv, 
    transforms=get_valid_transforms(), test=True, ROOT_PATH=VALID_ROOT_PATH
)

test_dataset = ODDataset(
    image_ids=test_df.index.values, marking=test_csv, 
    transforms=get_valid_transforms(), test=True, ROOT_PATH=TEST_ROOT_PATH
)

viz_train_augment(train_dataset)