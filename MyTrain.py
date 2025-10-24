import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.CTFNet import CTFNet
from utils.dataloader import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
from utils.dataloader import test_dataset
from utils.utils import eval_dice_iou
import numpy as np
import logging
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
best_performance = 0.0
best_epoch = 0
def structure_loss(pred, mask, smooth=1.):
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)).sum(dim=(2, 3))
    union = ((pred + mask)).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)   
    dice = (1 - ((2. * inter + smooth) / (pred.sum(dim=(2,3)) + mask.sum(dim=(2,3)) + smooth)))
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    return (wiou+dice).mean()
    #return (wiou+wbce+loss).mean()
def set_logging(config):
    if not os.path.exists(config.log_path): 
        os.makedirs(config.log_path)
    filename = os.path.join(config.log_path, config.log_file)
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )

def train_epoch(train_loader, model, optimizer, epoch):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record1, loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1 = model(images)
            # ---- loss function ----
            loss5 = structure_loss(lateral_map_5, gts)
            loss4 = structure_loss(lateral_map_4, gts)
            loss3 = structure_loss(lateral_map_3, gts)
            loss2 = structure_loss(lateral_map_2, gts)
            loss1 = structure_loss(lateral_map_1, gts)
            loss = loss1 + loss2 + loss3 + loss4 + loss5    # TODO: try different weights for loss
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record1.update(loss1.data, opt.batchsize)
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record3.update(loss3.data, opt.batchsize)
                loss_record4.update(loss4.data, opt.batchsize)
                loss_record5.update(loss5.data, opt.batchsize)
        # ---- train visualization ----
        if (i+1) % 10 == 0:
            logging.info('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-1: {:.4f}, lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record1.show(), loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()))

def eval_epoch(model, epoch, test_data_path):
    model.eval()
    global best_performance,best_epoch
    logging.info("######################################################")
    result_dice = []
    result_iou = []
    for _data_name in ['test']:
        data_path = os.path.join(test_data_path, _data_name)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        test_loader = test_dataset(image_root, gt_root, opt.testsize)
        dice = []
        iou = []
        
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            image = image.cuda()
            gt = gt.cuda()
            gt = gt.squeeze(0)
            res5, res4, res3, res2, res1 = model(image)
            # import torchvision.transforms as transform
            # res = transform.Resize(h,w)(res.squeeze(0))
            res = res1
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            gt = gt.unsqueeze(0)
            gt = gt.unsqueeze(0)
            
            
            dice_tmp, iou_tmp = eval_dice_iou(res, gt)
            # print(dice_tmp, iou_tmp)
            dice.append(dice_tmp.data.cpu().numpy())
            iou.append(iou_tmp.data.cpu().numpy())
        logging.info("eval epoch:{}, dataset:{}, mean_dice:{:.4f}, mean_iou:{:.4f}".format(epoch,_data_name,np.array(dice).mean(),np.array(iou).mean()))
        result_dice.append(np.array(dice).mean())
        result_iou.append(np.array(iou).mean())
        
    logging.info("eval epoch:{}, mean_dice:{:.4f}, mean_iou:{:.4f}".format(epoch,np.array(result_dice).mean(),np.array(result_iou).mean()))
    logging.info("######################################################")
    if np.array(result_dice).mean() + np.array(result_iou).mean() > best_performance:
        best_performance = np.array(result_dice).mean() + np.array(result_iou).mean()
        best_epoch = epoch
        #save weights
        save_path = './checkpoint/{}/'.format(opt.train_save)
        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), save_path + 'best_model.pth')
    logging.info("best epoch:{}".format(best_epoch))
    

    
if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=256, help='training dataset size')
    parser.add_argument('--testsize', type=int,
                        default=256, help='test dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,
                        default='./dataset/DDTI/train', help='path to train dataset')
    parser.add_argument('--test_path', type=str,
                        default='./dataset/DDTI/', help='path to test dataset')
    parser.add_argument('--log_path', type=str,
                        default='./log', help='path to log file')
    parser.add_argument('--log_file', type=str,
                        default='CTFNet-DDTI-Dataset-256.log', help='name to log file')
    parser.add_argument('--train_save', type=str,
                        default='CTFNet-DDTI-256')
    opt = parser.parse_args()
    set_logging(opt)
    logging.info(opt)
    # ---- build models ----
    torch.cuda.set_device(0)  # set your gpu device
    model = CTFNet().cuda()
    # model = torch.nn.DataParallel(model)
    # ---- flops and params ----
    # from utils.utils import CalParams
    # x = torch.randn(1, 3, 352, 352).cuda()
    # CalParams(lib, x)

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    # logging.info("#"*20, "Start Training", "#"*20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train_epoch(train_loader, model, optimizer, epoch)
        eval_epoch(model, epoch, opt.test_path)

