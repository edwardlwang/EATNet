import imageio
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from net.eatnet import Net
from utils.tdataloader import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=416, help='testing size')
parser.add_argument('--pth_path', type=str, default='./checkpoints/EATNet1/EATNet-24.pth')

for _data_name in ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']:
    data_path = './data/TestDataset/{}/'.format(_data_name)
    save_path = './results/EATNet/{}/'.format(_data_name)
    opt = parser.parse_args()
    model = Net()
    state_dict = torch.load(opt.pth_path)
    # 用 strict=False 加载，但自己把关：缺参数（结构不匹配）必须报错，
    # 多参数（如已删除的死参数）只警告，这样旧 checkpoint 仍可复现。
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError('权重缺少以下参数，拒绝加载：{}'.format(missing))
    if unexpected:
        print('[warn] checkpoint 含本模型已弃用的参数，已忽略：{}'.format(unexpected))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path + 'edge/', exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        _, _, res, e = model(image)
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        imageio.imwrite(save_path + name, (res * 255).astype(np.uint8))
        # e = F.upsample(e, size=gt.shape, mode='bilinear', align_corners=True)
        # e = e.data.cpu().numpy().squeeze()
        # e = (e - e.min()) / (e.max() - e.min() + 1e-8)
        # imageio.imwrite(save_path+'edge/'+name, (e*255).astype(np.uint8))
