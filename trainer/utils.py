import numpy as np
import matplotlib.pyplot as plt
import cv2

from helpers import class_dict as class_dict

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def show_image(labels, boxes, image, step, title, orig_dims=None):
    global class_dict

    image = image.squeeze(0)
    image = image.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image_np = np.array(image, dtype=np.float32)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    if title == 'Prediction':
        for i, box in enumerate(boxes):
            print(box)
            cv2.rectangle(
                image_np,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (0, 0, 255), 2
            )
            class_labels = class_dict[int(labels[i])]
            split_labels = class_labels.split()
            final_labels = ''.join(split_labels)
            cv2.putText(image_np, final_labels, (int(box[0]), int(box[1]-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, 
                    lineType=cv2.LINE_AA)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        plt.imshow(image_np)
        plt.title(f"{title}_{step}")
        plt.savefig(f"validation_image_results/{title}_{step}")
        # plt.show()
        plt.close()

    if title == 'Ground Truth':
        orig_dims = orig_dims
        for i, box in enumerate(boxes):
            # box[1] *= orig_dims[0]
            # box[0] *= orig_dims[1]
            # box[3] *= orig_dims[0]
            # box[2] *= orig_dims[1]
            # print(box)
            cv2.rectangle(
                image_np,
                (int(box[1]), int(box[0])),
                (int(box[3]), int(box[2])),
                (0, 0, 255), 2
            )
            class_labels = class_dict[int(labels[0][0][i])]
            split_labels = class_labels.split()
            final_labels = ''.join(split_labels)
            cv2.putText(image_np, final_labels, (int(box[1]), int(box[0]-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, 
                    lineType=cv2.LINE_AA)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        plt.imshow(image_np)
        plt.title(f"{title}_{step}")
        plt.savefig(f"validation_image_results/{title}_{step}")
        # plt.show()
        plt.close()
    

def get_output_boxes(outputs, step, image, threshold, save_results=None):
    # print('----------------------------')
    bboxes = outputs[0][:, :4]
    scores = list(outputs[0][:, 4])
    labels = list(outputs[0][:, 5])

    # consider boxes only above a certain threshold
    boxes = bboxes[np.array(scores) >= threshold]

    if save_results:
        show_image(labels, boxes, image, step, title='Prediction')
        
    with open(f"detection_results/{step}.txt", 'w') as f:
        for i in range(len(boxes)):
            class_labels = class_dict[int(labels[i])]
            split_labels = class_labels.split()
            final_labels = ''.join(split_labels)
            f.writelines(f"{final_labels} {scores[i]} {boxes[i][0]} {boxes[i][1]} {boxes[i][2]} {boxes[i][3]}\n")


def get_input_boxes(orig_dims, labels, boxes, step, image, save_results=None):
    """
    NOTE: Ground truth boxes are in yxyx format but the detections are in 
    xyxy format. So, while validating, that is calculating AP or mAP, 
    adjust either of the one to the other one's format. While creating the
    text file here, I have changed the ground truth coordinates to match the 
    prediction coordinates, i.e, from yxyx to xyxy
    """

    if save_results:
        show_image(labels, boxes[0][0], image, step, title='Ground Truth', orig_dims=orig_dims)

    with open(f"ground_truth/{step}.txt", 'w') as f:
        for i in range(len(labels[0][0])):
            # from yxyx that is 0123 to xyxy that is 1032
            class_labels = class_dict[int(labels[0][0][i])]
            split_labels = class_labels.split()
            final_labels = ''.join(split_labels)
            f.writelines(f"{final_labels} {boxes[0][0][i][1]} {boxes[0][0][i][0]} {boxes[0][0][i][3]} {boxes[0][0][i][2]}\n")