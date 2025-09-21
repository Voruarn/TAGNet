import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os, argparse
import imageio
from network.TagNet import TagNet
from setting.VLdataLoader import test_dataset
from tqdm import tqdm
import time
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument("--test_path", type=str, 
        default='../Datasets/RGB-DSOD/RGB-DSOD/', 
        help='Name of dataset')
parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument("--model", type=str, default='TagNet',
        help='model name:[TagNet]')
parser.add_argument('--convnext_model', type=str, default='convnext_base', 
                    help='ConvNext backbone: [convnext_base]')
parser.add_argument("--smap_save", type=str, default='../Sal_Preds/', help='save_path name')
parser.add_argument("--load", type=str,
            default="",
              help="restore from checkpoint")
opt = parser.parse_args()

def create_folder(save_path):
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Create Folder [“{save_path}”].")
    return save_path

model = eval(opt.model)(convnext_model_name=opt.convnext_model)

if opt.load is not None and os.path.isfile(opt.load):
    checkpoint = torch.load(opt.load, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)

    print("Model restored from %s" % opt.load)
    
model.cuda()
model.eval()

test_datasets = ['DUT-RGBD-Test', 'NJUD', 'NLPR', 'SIP', 'STERE']

for dataset in test_datasets:
    # load data
    data_path  = opt.test_path + dataset
    img_path = os.path.join(data_path, 'test_images/')
    depth_path = os.path.join(data_path, 'test_depth/')
    mask_path = os.path.join(data_path, 'test_masks/')
    text_path = os.path.join(data_path, 'test_text/')

    test_loader = test_dataset(img_path, depth_path, mask_path, text_path, opt.testsize)
    method=opt.load.split('/')[-1].split('.')[0]
    save_path = create_folder(opt.smap_save + dataset + '/'+method+'/')
    print('{} preds for {}'.format(method, dataset))
   
    cost_time = list()
    for i in tqdm(range(test_loader.size), desc=dataset):
        with torch.no_grad():
            image, depth, text, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            name = name.split('/')[-1]
            image = image.cuda()
            depth = depth.cuda()
            start_time = time.perf_counter()
            outputs = model(image, depth, text)
            res = outputs['sal_map']
            cost_time.append(time.perf_counter() - start_time)
            # res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(save_path+name, res*255)

    cost_time.pop(0)
    print('Mean running time is: ', np.mean(cost_time))
    print("FPS is: ", test_loader.size / np.sum(cost_time))
print("Test Done!")


