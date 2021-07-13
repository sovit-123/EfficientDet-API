import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib

from tensorboardX import SummaryWriter
from torch._C import dtype

class_dict = {
    1: 'boat'
}

class TensorboardWriter():
    def __init__(self):
        super(TensorboardWriter, self).__init__()
    # initilaize `SummaryWriter()`
        self.writer = SummaryWriter()
    def tensorboard_writer(
        self, loss=None, class_loss=None, box_loss=None, 
        iters=None, phase=None
    ):
        if phase == 'train':
            self.writer.add_scalar('Train Loss', loss, iters)
            self.writer.add_scalar('Train Class Loss', class_loss, iters)
            self.writer.add_scalar('Train Box Loss', box_loss, iters)
        if phase == 'valid':
            self.writer.add_scalar('Valid Loss', loss, iters)
            self.writer.add_scalar('Valid Class Loss', class_loss, iters)
            self.writer.add_scalar('Valid Box Loss', box_loss, iters)

def viz_train_augment(train_dataset):
    for i in range(10):
        # print(train_dataset[i])
        image, target, image_id = train_dataset[(i)]
        try:   
            boxes = target['boxes'].cpu().numpy()
        except:
            boxes = target['boxes']

        labels = target['labels']

        try:
            numpy_image = image.permute(1,2,0).cpu().numpy()
        except:
            # numpy_image = np.transpose(image, (1, 2, 0))
            numpy_image = image
            numpy_image = np.array(image, dtype=np.float32)

        fig, ax = plt.subplots(1, 1, figsize=(16, 8))

        for j, box in enumerate(boxes):
            """
            Uncomment the following lines if you have normalized bounding boxes
            """
            # box[1] *= target['dims'][0][0].astype(np.int32)
            # box[0] *= target['dims'][0][1].astype(np.int32)
            # box[3] *= target['dims'][0][0].astype(np.int32)
            # box[2] *= target['dims'][0][1].astype(np.int32)
            cv2.rectangle(
                numpy_image, (box[1], box[0]), (box[3],  box[2]), (0, 1, 0), 2 # if yxyx
                # numpy_image, (box[0], box[1]), (box[2],  box[3]), (0, 1, 0), 2 # if xyxy
            )
            # if yxyx #
            cv2.putText(numpy_image, class_dict[int(labels[j])], (int(box[1]), int(box[0]-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, 
                        lineType=cv2.LINE_AA)
            ##########

            # if xyxy #
            # cv2.putText(numpy_image, class_dict[int(labels[j])], (int(box[0]), int(box[1]-10)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, 
            #             lineType=cv2.LINE_AA)
            ###########

        ax.set_axis_off()
        ax.imshow(numpy_image)
        plt.savefig(f"aug_sample_images/image_{i}.jpg")
        # plt.show()
        plt.close()

def viz_valid_augment(valid_dataset):
    # visualize some validation augmented images
    for i in range(10):
        image, target, image_id = valid_dataset[i]

        boxes = target['boxes'].cpu().numpy().astype(np.int32)

        numpy_image = image.permute(1,2,0).cpu().numpy()

        fig, ax = plt.subplots(1, 1, figsize=(16, 8))

        for box in boxes:
            cv2.rectangle(numpy_image, (box[1], box[0]), (box[3],  box[2]), (0, 1, 0), 2)

        ax.set_axis_off()
        ax.imshow(numpy_image)
        # plt.show()
        plt.close()