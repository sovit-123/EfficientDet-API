import sys

sys.path.insert(0, '/home/sovit/my_data/Data_Science/cloned_gits/efficientdet-pytorch')

import torch
import time

from utils import get_input_boxes, get_output_boxes
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from dataset import test_dataset
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict


test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1,
        # num_workers=TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(test_dataset),
        pin_memory=False,
    )

def get_net():
    config = get_efficientdet_config('efficientdet_d0')
    config.num_classes = 1
    config.image_size = [512, 512]
    net = EfficientDet(config, pretrained_backbone=False)
    checkpoint = torch.load('effdet_checkpoints/2021-04-14 07_57_36/best-checkpoint-044epoch.pth')
    net.load_state_dict(checkpoint['model_state_dict'])
    return DetBenchPredict(net)

model = get_net()

device = 'cuda'

print(len(test_loader))

threshold = 0.3
save_results = True

def evaluation(test_dataset, test_loader, model, device):
        model.eval()
        model.to(device)
        t = time.time()

        for step, (images, targets, image_ids) in enumerate(test_loader):
            print(f"INFO: Step {step}")
            with torch.no_grad():
                # images = torch.stack(images)
                images = images
                batch_size = images.shape[0]
                images = images.to(device)
                boxes = [targets['boxes'].to(device).float()]
                labels = [targets['labels'].to(device).float()]

                ##### EXTRA LINES FOR CORRECT VALIDATION #####
                target_res = {}
                target_res['bbox'] = boxes
                target_res['cls'] = labels
                target_res['img_scale'] = torch.tensor([1.0] * batch_size, dtype=torch.float).to(device)
                target_res['img_size'] = torch.tensor([images[0].shape[-2:]] * batch_size, dtype=torch.float).to(device)
                ##############################################

                # FEW CHANGED LINES OF CODE
                outputs = model(images)
                get_output_boxes(outputs, step, images, threshold=threshold, save_results=save_results)
                get_input_boxes(test_dataset[step][1]['dims'][0], target_res['cls'], target_res['bbox'], step, images, save_results=save_results)

                # CALLING MY CUSTOM FUNCTIONS
                # draw_box(images[0], outputs['detections'][0])
                #################################

evaluation(test_dataset, test_loader, model, device)