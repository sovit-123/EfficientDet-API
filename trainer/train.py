from engine import run_training
import sys
import argparse
import torch
import random
import os
import numpy as np

sys.path.insert(0, '/home/sovit/my_data/Data_Science/cloned_gits/efficientdet-pytorch')

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from engine import run_training

SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

parser = argparse.ArgumentParser()
parser.add_argument(
    '-r', '--resume-training', dest='resume_training', default='no', type=str,
    choices=['yes', 'no'], help='whether to resume training or not'
)
parser.add_argument(
    '-c', '--checkpoint', required=False, type=str,
    help='path to checkpoint to continue training'
)
args = vars(parser.parse_args())

def get_net(model_name):
    config = get_efficientdet_config(model_name)
    config.norm_kwargs=dict(eps=.001, momentum=.01)
    config.num_classes = 1
    # config.image_size = [640, 640]
    net = EfficientDet(config, pretrained_backbone=True)
    
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    return DetBenchTrain(net, config)

model_name = 'efficientdet_d0'
net = get_net(model_name)
run_training(net, args, model_name=model_name)