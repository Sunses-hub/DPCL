
# evaluation code


import os
from datetime import datetime
from utils import *
import random
from network.unet2d import UNet2D
from dataset.chd import CHD
from dataset.acdc import ACDC
from dataset.mmwhs import MMWHS
from dataset.hvsmr import HVSMR
import torch.nn.functional as F
from metrics import SegmentationMetric
from myconfig import get_config
from batchgenerators.utilities.file_and_folder_operations import *
from lr_scheduler import LR_Scheduler
from torch.utils.tensorboard import SummaryWriter
from experiment_log import PytorchExperimentLogger


def run(fold, writer, args):
    maybe_mkdir_p(os.path.join(args.save_path, 'cross_val_' + str(fold)))
    logger = PytorchExperimentLogger(os.path.join(args.save_path, 'cross_val_' + str(fold)), "elog", ShowTerminal=True)
    # setup cuda
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.print(f"the model will run on device:{args.device}")
    torch.manual_seed(args.seed)
    if 'cuda' in str(args.device):
        torch.cuda.manual_seed_all(args.seed)
    logger.print(f"starting training for cross validation fold {fold} ...")
    model_result_dir = join(args.save_path, 'cross_val_' + str(fold), 'model')
    maybe_mkdir_p(model_result_dir)
    args.model_result_dir = model_result_dir
    # create model
    logger.print("creating model ...")
    model = UNet2D(in_channels=1, initial_filter_size=args.initial_filter_size, kernel_size=3, classes=args.classes,
                   do_instancenorm=True)

    '''
    if args.restart:
        logger.print('loading from saved model ' + args.pretrained_model_path)
        dict = torch.load(args.pretrained_model_path,
                          map_location=lambda storage, loc: storage)
        save_model = dict["net"]
        model_dict = model.state_dict()
        # we only need to load the parameters of the encoder
        state_dict = {k: v for k, v in save_model.items() if "encoder" in k}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    model.to(args.device)
    '''

    if args.evaluation:
        logger.print('loading from saved model ' + args.pretrained_model_path)
        dict = torch.load(args.pretrained_model_path,
                          map_location=lambda storage, loc: storage)
        save_model = dict["net"]
        model_dict = model.state_dict()
        # we only need to load the parameters of the encoder
        state_dict = {k: v for k, v in save_model.items() if "encoder" in k}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    model.to(args.device)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logger.print(f"number of parameters: {num_parameters}")

    train_keys, val_keys = get_split_acdc(fold, args.cross_vali_num)
    if args.enable_few_data:
        random.seed(args.seed)
        train_keys = random.sample(list(train_keys), k=args.sampling_k)
    #logger.print(f'train_keys:{train_keys}')
    logger.print(f'val_keys:{val_keys}')
    #train_dataset = ACDC(keys=train_keys, purpose='train', args=args)
    validate_dataset = ACDC(keys=val_keys, purpose='val', args=args)

    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
    #                                           num_workers=args.num_works, drop_last=False)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_works, drop_last=False)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5)
    #scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(train_loader), min_lr=args.min_lr)
    best_dice = 0
    for epoch in range(args.epochs):
        # evaluate for one epoch
        val_dice = validate(validate_loader, model, epoch, logger, args)

        if best_dice < val_dice:
            best_dice = val_dice
            save_dict = {"net": model.state_dict()}
            torch.save(save_dict, os.path.join(args.model_result_dir, "best.pth"))
        writer.add_scalar('validate_dice_fold' + str(fold), val_dice, epoch)
        writer.add_scalar('best_dice_fold' + str(fold), best_dice, epoch)
        # save model
        save_dict = {"net": model.state_dict()}
        torch.save(save_dict, os.path.join(args.model_result_dir, "latest.pth"))


def validate(data_loader, model, epoch, logger, args):
    model.eval()
    metric_val = SegmentationMetric(args.classes)
    metric_val.reset()
    with torch.no_grad():
        for batch_idx, tup in enumerate(data_loader):
            img, label = tup
            image_var = img.float().to(args.device)
            label = label.long().to(args.device)
            x_out = model(image_var)
            x_out = F.softmax(x_out, dim=1)
            metric_val.update(label.long().squeeze(dim=1), x_out)
            pixAcc, mIoU, Dice = metric_val.get()
            logger.print(f"Validation epoch:{epoch}, batch:{batch_idx}/{len(data_loader)}, mean Dice:{Dice}")
    pixAcc, mIoU, Dice = metric_val.get()
    return Dice


if __name__ == '__main__':
    # initialize config
    args = get_config()
    if args.save == '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.save_path = os.path.join(args.results_dir, args.experiment_name + args.save)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    writer = SummaryWriter(os.path.join(args.runs_dir, args.experiment_name + args.save))
    for i in range(0, args.cross_vali_num):
        run(i, writer, args)