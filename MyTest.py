import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime

from utils.dataloader import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
from utils.dataloader import test_dataset
import numpy as np
import logging
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
best_performance = 0.0
best_epoch = 0
def eval_dice_iou(pred, mask, smooth=1.):
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)).sum(dim=(2, 3))
    union = ((pred + mask)).sum(dim=(2, 3))
    iou = (inter + 1)/(union - inter+1)
    dice =  ((2. * inter + smooth) / (pred.sum(dim=(2,3)) + mask.sum(dim=(2,3)) + smooth))
    return dice,iou

def eval_epoch(model, test_data_path):
    model.eval()
    for _data_name in ['test']:
        data_path = os.path.join(test_data_path, _data_name)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        test_loader = test_dataset(image_root, gt_root, opt.testsize)
        dice = []
        iou = []
        mae = []
        hd95 = []
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            image = image.cuda()
            gt = gt.cuda()
            gt = gt.squeeze(0)
            res5, res4, res3, res2, res1 = model(image)
            res = res1
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            mask = res.cpu().detach().numpy()[0][0] * 255
            # print(case)
            import cv2
            save_path = opt.save_path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imencode('.png', mask)[1].tofile(save_path + name)
            print(save_path + name)

            gt = gt.unsqueeze(0)
            gt = gt.unsqueeze(0)
            
            
            dice_tmp, iou_tmp = eval_dice_iou(res, gt)
            # print(dice_tmp, iou_tmp)
            dice.append(dice_tmp.data.cpu().numpy())
            iou.append(iou_tmp.data.cpu().numpy())
        print("mean_dice:{:.4f}, mean_iou:{:.4f}".format(np.array(dice).mean(),np.array(iou).mean()))
        # print("mean_mae:{:.4f}, mean_hd95:{:.4f}".format(np.array(mae).mean(),np.array(hd95).mean()))
        
    

    
if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int,
                        default=256, help='test dataset size')
    parser.add_argument('--test_path', type=str,
                        default='./dataset/CVC-ClinicDB', help='path to test dataset')
                                        # DDTI,TNSCUI,TN3K,HMUTND
                                        # CVC-ClinicDB, CVC-ColonDB,
                                        # ETIS, Kvasir-SEG
    parser.add_argument('--model_path', type=str,                                                                       
                        #default='/home/bucea/LWF/save_checkpoint/CTF-Net-Experiments/CTF-Net/checkpoint/CTFNet-4stage-DDTI-256/best_model.pth')
                        default='/home/bucea/LWF/save_checkpoint/CTF-Net-Experiments/CTF-Net/checkpoint/CTFNet-4stage-DDTI-256/best_model.pth')
    parser.add_argument('--save_path', type=str,
                        #default='./predict_result/')
                        default='./bst_predict_result3/')
    from lib.CTFNet import CTFNet
    opt = parser.parse_args()
    # ---- build models ----
    torch.cuda.set_device(0)  # set your gpu device
    model = CTFNet().cuda()
    model.load_state_dict(torch.load(opt.model_path, map_location='cuda:0'))
    eval_epoch(model, opt.test_path)

