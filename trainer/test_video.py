"""
USAGE:
python test_video.py -i ../data/inference_data/video_1.mp4 -c "effdet_checkpoints/2021-04-14 07_57_36/best-checkpoint-044epoch.pth"
"""

import torch 
import numpy as np
import argparse
import cv2
import albumentations as A
import time
import sys

from albumentations.pytorch.transforms import ToTensorV2

sys.path.insert(0, '/home/sovit/my_data/Data_Science/cloned_gits/efficientdet-pytorch')

from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
from effdet.efficientdet import HeadNet

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to the input video', 
                    required=True)
parser.add_argument(
    '-c', '--checkpoint', required=False, type=str,
    help='path to checkpoint to continue training'
)
args = vars(parser.parse_args())

file_path = args['input']
cap = cv2.VideoCapture(file_path)
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')

# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

print(f"[INFO]: Video's original frame width and height: {frame_width}, {frame_height}")

w, h = 512, 512 # potentially, we can give any size (the code will take care of the rest)
threshold = 0.37
print(f"[INFO]: Reszied to: {w}x{h}")
print(f"[INFO]: Threshold {threshold}")

def get_net(model_name):
    config = get_efficientdet_config(model_name)
    # config.norm_kwargs=dict(eps=.001, momentum=.01)
    config.num_classes = 1
    # by default the interal resize is 512x512 (hxw)
    config.image_size = [h, w]
    net = EfficientDet(config, pretrained_backbone=False)
    # net.reset_head(num_classes=3)
    path = args['checkpoint']
    print('[INFO]: Model path:', path)
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['model_state_dict'])

    # most probably we do not need the following line
    # net.class_net = HeadNet(config, num_outputs=config.num_classes)
    return DetBenchPredict(net)

model_name = 'efficientdet_d0'
net = get_net(model_name)

# /content/drive/MyDrive/Data Science/Mirrag/small_large_od_tests/dump_hopp_helmet_combined/inference

save_name = f"../../outputs/{file_path.split('/')[-1].split('.')[0]}_{model_name}_{int(threshold*100)}_{w}"
# define codec and create VideoWriter object 
out = cv2.VideoWriter(f"{save_name}_pt.avi", 
                      cv2.VideoWriter_fourcc(*'MJPG'), 15, 
                      (frame_width, frame_height))

resize_transform = A.Compose([
               A.Resize(h, w),
])
transform = A.Compose([
            ToTensorV2(p=1.0)], p=1.0)

classes = {
    1: 'boat',
}

colors = {
    'boat': (0, 255, 0)
}

num_detected_boats = 0

def draw_bboxes(image, outputs, w, h):

    global num_detected_boats

    orig_h, orig_w = image.shape[0], image.shape[1]

    scores = list(outputs[0][:, 4])
    # threshold_indices = [scores.index(i) for i in scores if i > 0.2]
    labels = list(outputs[0][:, 5])
    
    bboxes = outputs[0][:, :4].detach().cpu().numpy()

    boxes = bboxes[np.array(scores) >= threshold].astype(np.int32)

    # print(boxes)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for i, box in enumerate(boxes):
        box_0 = ((box[0]/w)*orig_w)
        box_1 = ((box[1]/h)*orig_h)
        box_2 = ((box[2]/w)*orig_w)
        box_3 = ((box[3]/h)*orig_h)
        cv2.rectangle(
            image,
            (int(box_0), int(box_1)),
            (int(box_2), int(box_3)),
            colors[classes[int(labels[i])]], 2
        )
        cv2.putText(image, classes[int(labels[i])], (int(box_0), int(box_1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[classes[int(labels[i])]], 2, 
                    lineType=cv2.LINE_AA)
        if labels[i] == 1:
            num_detected_boats += 1
    return image

frame_count = 0 # to count total frames
total_fps = 0 # to get the final frames per second

net.eval()
net = net.cuda()
# read until end of video
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        orig_copy = frame.copy()
        frame = resize_transform(image=frame)['image']
        orig_frame = frame.copy()
        frame = np.array(frame, dtype=np.float32)
        frame /= 255.0

        frame = transform(image=np.array(frame))['image']

        input_tensor = np.expand_dims(frame, 0)
        input_tensor = torch.tensor(input_tensor, dtype=torch.float)
        input_tensor = input_tensor.cuda()
        # get the start time
        start_time = time.time()
        with torch.no_grad():
            # get the detections
            outputs = net(input_tensor)

        result = draw_bboxes(orig_copy, outputs, w, h)

        # get the end time
        end_time = time.time()
        # get the fps
        fps = 1 / (end_time - start_time)
        # add fps to total fps
        total_fps += fps
        # increment frame count
        frame_count += 1

        # press `q` to exit
        wait_time = max(1, int(fps/4))
        
        cv2.putText(
            result, f"{fps:.3f} FPS",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0, 0, 255), 2, cv2.LINE_AA
        )
        cv2.imshow('Result', result)

        # out.write(result)
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    else:
        break

# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()

# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")

print('Number of boats detected: ', num_detected_boats)