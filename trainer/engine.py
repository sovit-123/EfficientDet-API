import warnings
import time
import torch
import os
import glob as glob
import datetime

from datetime import datetime
from utils import AverageMeter
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from dataset import train_dataset, valid_dataset
from helpers import TensorboardWriter


warnings.filterwarnings("ignore")

class Fitter:
    def __init__(self, args, model, device, config, model_name):
        self.model_name = model_name

        self.args = args
        self.config = config
        self.device = device
        self.model = model

        # initialize Tensorboard `SummaryWriter()`
        self.writer = TensorboardWriter()

        # ADAMW OPTIMIZER
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        # SGD WITH MOMENTUM OPTIMIZER
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.lr, momentum=0.9)
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 

        if self.args['resume_training'] == 'no':
            self.base_dir = f'./{config.folder}'
            if not os.path.exists(self.base_dir):
                os.makedirs(self.base_dir)
            self.log_path = f'{self.base_dir}/log.txt'

            self.log('INFO: Training from beginning')
            self.epoch = 0
            self.epochs_to_train = self.config.n_epochs
            # initialize training and validation steps
            self.train_steps = 0
            self.valid_steps = 0
            
            self.best_summary_loss = 10**5

            self.log('INFO: Model, optimizer, and config initialization done')
            self.log(f'Fitter prepared. Device is {self.device}')
            self.log(f'INFO: Model name = {self.model_name}')

        elif self.args['resume_training'] == 'yes':
            self.base_dir = f'./{config.folder}'
            if not os.path.exists(self.base_dir):
                os.makedirs(self.base_dir)
            self.log_path = f'{self.base_dir}/log.txt'

            assert(self.args['checkpoint'] != None), 'Please provide a path to resume training'

            checkpoint = torch.load(self.args['checkpoint'])
            self.model.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_summary_loss = checkpoint['best_summary_loss']
            self.epoch = checkpoint['epoch'] + 1
            self.train_steps = checkpoint['train_steps'] + 1
            self.valid_steps = checkpoint['valid_steps'] + 1

            self.epochs_to_train = self.config.n_epochs - (self.epoch)

            assert(self.config.n_epochs > self.epoch), f"Please provide greater than {self.config.n_epochs} epochs to train"

            self.log('INFO: Model checkpoint loaded')
            self.log('INFO: Optimizer state dict loaded')
            self.log('INFO: Learning rate scheduler state dict loaded')
            self.log('INFO: Checkpoint summary loss loaded')
            self.log(f"INFO: Resuming training from epoch {self.epoch}")
            self.log(f"INFO: Resuming training from {self.train_steps} training steps")
            self.log(f"INFO: Resuming training from {self.valid_steps} validation steps")

            self.log(f'Fitter prepared. Device is {self.device}')
            self.log(f'INFO: Model name = {self.model_name}')

    def fit(self, train_loader, validation_loader):
        for e in range(self.epochs_to_train):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            train_summary_loss, train_class_loss, train_box_loss = self.train_one_epoch(train_loader, self.scheduler, self.epoch)

            # TENSORBOARD TRAIN LOGGING BEGINS #
            self.writer.tensorboard_writer(
                loss=train_summary_loss.avg,
                class_loss=train_class_loss.avg,
                box_loss=train_box_loss.avg,
                iters=self.epoch, phase='train'
            )
            # TENSORBOARD VALIDATION LOGGING ENDS #

            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {train_summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            # self.save(f'{self.base_dir}/last-checkpoint.pth')

            t = time.time()
            valid_summary_loss, valid_class_loss, valid_box_loss = self.validation(validation_loader)

            # TENSORBOARD VALIDATION LOGGING BEGINS #
            self.writer.tensorboard_writer(
                loss=valid_summary_loss.avg,
                class_loss=valid_class_loss.avg, 
                box_loss=valid_box_loss.avg, 
                iters=self.epoch, phase='valid'
            )
            # TENSORBOARD VALIDATION LOGGING ENDS #

            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {valid_summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            # save the latest and the best model
            self.save(f'{self.base_dir}/last-checkpoint.pth')
            if valid_summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = valid_summary_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.pth')
                for path in sorted(glob.glob(f'{self.base_dir}/best-checkpoint-*epoch.pth'))[:-3]:
                    os.remove(path)

            """
            Double check everything before using this scheduler. It is to be 
            scheduled with average validation loss and not training loss.
            """
            if self.config.validation_scheduler:
                # self.log(f'[INFO]: Validation scheduler step')
                self.scheduler.step(metrics=valid_summary_loss.avg)

            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        epoch_class_loss = AverageMeter()
        epoch_box_loss = AverageMeter()
        t = time.time()

        for step, (images, targets, image_ids) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                images = torch.stack(images)
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                boxes = [target['boxes'].to(self.device).float() for target in targets]
                labels = [target['labels'].to(self.device).float() for target in targets]

                ##### EXTRA LINES FOR CORRECT VALIDATION #####
                target_res = {}
                target_res['bbox'] = boxes
                target_res['cls'] = labels 
                target_res['img_scale'] = torch.tensor([1.0] * batch_size, dtype=torch.float).to(self.device)
                target_res['img_size'] = torch.tensor([images[0].shape[-2:]] * batch_size, dtype=torch.float).to(self.device)
                ##############################################

                # FEW CHANGED LINES OF CODE
                outputs = self.model(images, target_res)

                # CALLING MY CUSTOM FUNCTIONS
                # draw_box(images[0], outputs['detections'][0])
                #################################

                loss = outputs['loss']
                class_loss = outputs['class_loss']
                box_loss = outputs['box_loss']

                summary_loss.update(loss.detach().item(), batch_size)
                epoch_class_loss.update(class_loss.detach().item(), batch_size)
                epoch_box_loss.update(box_loss.detach().item(), batch_size)

                ##### TESORBOARD LOGGING #####
                # self.writer.tensorboard_writer(
                #     loss=outputs['loss'],
                #     class_loss=outputs['class_loss'], 
                #     box_loss=outputs['box_loss'],
                #     iters=self.valid_steps, phase='valid'
                # )
                ##############################

            self.valid_steps += 1

        return summary_loss, epoch_class_loss, epoch_box_loss

    def train_one_epoch(self, train_loader, scheduler, epoch):
        self.model.train()
        summary_loss = AverageMeter()
        epoch_class_loss = AverageMeter()
        epoch_box_loss = AverageMeter()

        # we need the following `iters` for Cosine Annealing with Warm Restart scheduling
        iters = len(train_loader)

        t = time.time()
        for step, (images, targets, image_ids) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )

            images = torch.stack(images)
            images = images.to(self.device).float()
            batch_size = images.shape[0]
            boxes = [target['boxes'].to(self.device).float() for target in targets]
            labels = [target['labels'].to(self.device).float() for target in targets]

            ##### EXTRA LINES FOR CORRECT TRAINING #####
            target_res = {} # ADDED THIS LINE! REALLY IMPORTANT
            target_res['bbox'] = boxes # ADDED THIS LINE! REALLY IMPORTANT
            target_res['cls'] = labels # ADDED THIS LINE! REALLY IMPORTANT
            ############################################

            self.optimizer.zero_grad()
            
            # changed `target` to `target_res` and also changed it a bit
            outputs = self.model(images, target_res)
            loss = outputs['loss']
            class_loss = outputs['class_loss']
            box_loss = outputs['box_loss']
            
            loss.backward()

            summary_loss.update(loss.detach().item(), batch_size)
            epoch_class_loss.update(class_loss.detach().item(), batch_size)
            epoch_box_loss.update(box_loss.detach().item(), batch_size)

            self.optimizer.step()

            """
            The following is the scheduling for Cosine Annealing with Warm
            Restarts. It should be after every batch iteration.
            """
            if self.config.cosine_annealing_scheduler:
                # self.log(f'[INFO]: Cosine Annealing scheduler step')
                scheduler.step(epoch + (step / iters))

            """
            The following is step scheduler after training
            """
            if self.config.step_scheduler:
                # self.log(f'[INFO]: Training scheduler step')
                self.scheduler.step()

            self.train_steps += 1

        return summary_loss, epoch_class_loss, epoch_box_loss
    
    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
            'train_steps': self.train_steps,
            'valid_steps': self.valid_steps,
        }, path)
        
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')

class TrainGlobalConfig:
    num_workers = 4
    batch_size = 8
    n_epochs = 50
    lr = 0.001

    time = datetime.now()
    time_split = str(time).split('.')[0].split(':')
    print(time_split)
    time_joined = '_'.join(time_split)
    folder = os.path.join('effdet_checkpoints', time_joined)

    # -------------------
    verbose = True
    verbose_step = 1
    # -------------------

    # --------------------

    """
    Currently all the scheduler names are the same `scheduler`. So, use accordingly 
    or change each name according to requirement.
    """
    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = False  # do scheduler.step after validation stage loss
    cosine_annealing_scheduler = False # do scheduler.step after optimizer.step

    # COSINE ANNEALING WITH WARM RESTARTS
    steps = n_epochs
    SchedulerClass = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    scheduler_params = dict(
        T_0=steps,
        # verbose=True
    )

#     SchedulerClass = torch.optim.lr_scheduler.OneCycleLR
#     scheduler_params = dict(
#         max_lr=0.001,
#         epochs=n_epochs,
#         steps_per_epoch=int(len(train_dataset) / batch_size),
#         pct_start=0.1,
#         anneal_strategy='cos', 
#         final_div_factor=10**5
#     )
    
    # SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    # scheduler_params = dict(
    #     mode='min',
    #     factor=0.5,
    #     patience=1,
    #     verbose=False, 
    #     threshold=0.0001,
    #     threshold_mode='abs',
    #     cooldown=0, 
    #     min_lr=1e-8,
    #     eps=1e-08
    # )
    # --------------------


def collate_fn(batch):
    return tuple(zip(*batch))

def run_training(net, args, model_name):
    device = torch.device('cuda:0')
    net.to(device)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=TrainGlobalConfig.batch_size,
        num_workers=TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(valid_dataset),
        pin_memory=False,
        collate_fn=collate_fn,
    )

    fitter = Fitter(
        args, model=net, device=device, config=TrainGlobalConfig,
        model_name=model_name
    )
    fitter.fit(train_loader, val_loader)