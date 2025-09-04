import torch
import numpy as np
import pdb, os, argparse
from datetime import datetime
import sys
from tqdm import tqdm
from setting.dataLoader import get_loader
from setting.utils import clip_gradient, adjust_lr
from network.SimSOD import SimSOD
import pytorch_iou
from metrics.SOD_metrics import SODMetrics
from torch.utils.tensorboard import SummaryWriter



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument("--trainset_path", type=str, 
        default='/fuyuxiang/Projects/CLS/Datasets/RGB-DSOD11/RGB-DSOD/RGBD_Train/')
parser.add_argument("--testset_path", type=str, 
        default='/fuyuxiang/Projects/CLS/Datasets/RGB-DSOD11/RGB-DSOD/DUT-RGBD-Test/')
parser.add_argument("--dataset", type=str, default='RGBD29K', 
                    help='Name of dataset:[RGBDSOD]')
parser.add_argument('--model', type=str, default='SimSOD', 
                    help='model name:[SimSOD]')
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')

parser.add_argument("--load", type=str,
            default='', help="restore from checkpoint")
parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
parser.add_argument("--n_cpu", type=int, default=8, help="num of workers")
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=1000, help='every n epochs decay learning rate')
parser.add_argument('--save_path', type=str, default='./CHKP/', help='')
parser.add_argument('--log_dir', type=str, default='./logs/', help='')
parser.add_argument('--save_ep', type=int, default=5, help='')
opt = parser.parse_args()


def validate(opts, model, loader, device,  metrics):
    metrics.reset()
    with torch.no_grad():
        for step, (images, depths, labels)  in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)
            depths = depths.cuda()

            sal, sal_sig = model(images, depths)
              
            outputs=sal_sig
            preds=outputs.squeeze()
            labels=labels.squeeze()
    
            metrics.update(preds, labels)

        score = metrics.get_results()
    return score

CE = torch.nn.BCEWithLogitsLoss()
IOU = pytorch_iou.IOU(size_average = True)


if __name__=='__main__':
    # build models
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    opt.log_dir=opt.log_dir+'{}_{}_ep{}_lr{}'.format(opt.model, opt.dataset, opt.epoch, str(opt.lr))
    tb_writer = SummaryWriter(opt.log_dir)

    print(opt)
    model = eval(opt.model)()

    if opt.load is not None and os.path.isfile(opt.load):
        checkpoint = torch.load(opt.load, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)

        print("Model restored from %s" % opt.load)
    
    model.cuda()
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)
    metrics=SODMetrics(cuda=True)
    # load data
    train_loader, train_num = get_loader(opt.trainset_path+'train_images/',
                                        opt.trainset_path+'train_depth/', 
                                        opt.trainset_path+'train_masks/', 
                                        opt.batchsize, opt.trainsize, num_workers=opt.n_cpu)
    val_loader, val_num = get_loader(opt.testset_path+'test_images/',
                                        opt.testset_path+'test_depth/', 
                                        opt.testset_path+'test_masks/', 
                                        opt.batchsize, opt.trainsize, num_workers=opt.n_cpu)
    print(f'Loading data, including {train_num} training images and {val_num} validation images.')

    print("Let's go!")
    cur_epoch=0

    for epoch in range(cur_epoch, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        model.train()
        running_loss = 0.0
        data_loader = tqdm(train_loader, file=sys.stdout)
        steps=0
        for i, (images, depths, gts) in enumerate(data_loader, start=1):
            steps+=1
            optimizer.zero_grad()
            images = images.cuda()
            depths = depths.cuda()
            gts = gts.cuda()

            sal, sal_sig = model(images, depths)
            loss = CE(sal, gts) + IOU(sal_sig, gts)

            running_loss += loss.data.item()
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            data_loader.desc = "Epoch {}/{}, Learning Rate: {}, loss={:.4f}".format(epoch, opt.epoch,
                                opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), running_loss / i)

        print("validation...")
        model.eval()
        val_score = validate(
            opts=opt, model=model, loader=val_loader, device=device, metrics=metrics)
        
        print('val_score:',val_score)
    
        tags = ["train_loss", "learning_rate","MAE","Sm"]

        tb_writer.add_scalar(tags[0], running_loss/steps, epoch)
        tb_writer.add_scalar(tags[1], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar(tags[2], val_score["MAE"], epoch)
        tb_writer.add_scalar(tags[3], val_score["Sm"], epoch)

        
        if (epoch+1) % opt.save_ep == 0:
            torch.save(model.state_dict(), opt.save_path+'lastest_{}_{}.pth'.format(opt.model, opt.dataset))
            if (epoch+1) % 25 == 0:
                torch.save(model.state_dict(), opt.save_path+'{}_{}_{}e.pth'.format(opt.model, opt.dataset, (epoch+1)))


