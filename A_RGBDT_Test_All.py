import torch
import numpy as np
import random
import os, argparse
from tqdm import tqdm
from setting.VLdataLoader import get_loader
from network.TagNetV3 import TagNet
from metrics.SOD_metrics import SODMetrics
from torch.utils.tensorboard import SummaryWriter


# 固定随机数种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--base_data_path", type=str, 
        default='/fuyuxiang/Projects/CLS/Datasets/RGB-DSOD11/RGB-DSOD/', 
        help="所有数据集的根目录")
parser.add_argument('--model', type=str, default='TagNet', 
                    help='模型名称')
parser.add_argument('--convnext_model', type=str, default='convnext_small', 
                    help='ConvNext backbone: [convnext_tiny, convnext_small, convnext_base]')
parser.add_argument("--load", type=str, default="/fuyuxiang/Projects/CLS/Seg/TagNet/CHKP0905_s/TagNet_small_RGBDT29K_100e.pth", help="模型权重文件路径")
parser.add_argument('--batchsize', type=int, default=16, help='测试批次大小')
parser.add_argument("--n_cpu", type=int, default=8, help="数据加载线程数")
parser.add_argument('--testsize', type=int, default=256, help='测试图像尺寸')
parser.add_argument('--log_dir', type=str, default='./test_logs/', help='日志保存路径')
opt = parser.parse_args()


# 定义所有需要测试的数据集（名称: 子路径）
DATASETS = {
    "DUT-RGBD-Test": "DUT-RGBD-Test/",
    "LFSD": "LFSD/",
    "NJUD": "NJUD/",
    "NLPR": "NLPR/",
    "SIP": "SIP/",
    "SSD": "SSD/",
    "STERE": "STERE/"
}


def validate(model, loader, device, metrics):
    """验证单个数据集"""
    metrics.reset()
    with torch.no_grad():
        for step, (images, depths, texts, labels) in tqdm(enumerate(loader), desc="测试中"):
            # 数据设备迁移
            images = images.to(device, dtype=torch.float32)
            depths = depths.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            # 模型推理
            outputs = model(images, depths, texts)
            sal_map = outputs['sal_map']  # 形状应为 (B, 1, H, W)

            # 关键修复：只移除通道维度（第1维），保留 (B, H, W)
            # 避免使用 squeeze() 盲目移除所有维度为1的维度（可能破坏空间结构）
            preds = sal_map.squeeze(1)  # (B, 1, H, W) → (B, H, W)
            labels = labels.squeeze(1)  # (B, 1, H, W) → (B, H, W)

            # 确保维度正确（必须是3维：B, H, W）
            assert len(preds.shape) == 3, f"预测结果维度错误：{preds.shape}，应为 (B, H, W)"
            assert len(labels.shape) == 3, f"标签维度错误：{labels.shape}，应为 (B, H, W)"

            metrics.update(preds, labels)

    return metrics.get_results()


def test_on_dataset(model, dataset_name, data_path, opt, metrics):
    """测试单个数据集并返回结果"""
    print(f"\n===== 开始测试数据集: {dataset_name} =====")
    
    # 构建数据集路径
    img_path = os.path.join(data_path, 'test_images/')
    depth_path = os.path.join(data_path, 'gen_depth/')
    mask_path = os.path.join(data_path, 'test_masks/')
    text_path = os.path.join(data_path, 'test_text_05b/')
    
    # 检查路径是否存在
    for path in [img_path, depth_path, mask_path, text_path]:
        if not os.path.exists(path):
            print(f"警告: 路径不存在 - {path}，跳过该数据集")
            return None
    
    # 创建数据加载器
    test_loader, test_num = get_loader(
        img_path,
        depth_path,
        mask_path,
        text_path,
        batchsize=opt.batchsize,
        trainsize=opt.testsize,
        num_workers=opt.n_cpu,
    )
    print(f"加载数据集 {dataset_name}，共 {test_num} 张图像")
    
    # 执行测试
    model.eval()
    results = validate(model, test_loader, device, metrics)
    
    # 打印当前数据集结果
    print(f"数据集 {dataset_name} 测试结果:")
    print(f"MAE: {results['MAE']:.4f}, Sm: {results['Sm']:.4f} " )
    return {dataset_name: results}


if __name__=='__main__':

    print(f"[配置信息] {opt}")

    # 2. 初始化模型
    model = TagNet(convnext_model_name=opt.convnext_model).to(device)
    print(f"[模型加载] 加载TagNet，backbone: {opt.convnext_model}")

    # 3. 加载预训练权重
    if os.path.isfile(opt.load):
        checkpoint = torch.load(opt.load, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"[权重加载] 从 {opt.load} 恢复模型权重")
    else:
        print(f"[错误] 权重文件不存在: {opt.load}")
        exit(1)

    # 4. 初始化评估指标
    metrics = SODMetrics(cuda=True)

    # 5. 批量测试所有数据集
    all_results = {}
    for dataset_name, subpath in DATASETS.items():
        # 构建完整数据集路径
        dataset_full_path = os.path.join(opt.base_data_path, subpath)
        # 测试单个数据集
        result = test_on_dataset(model, dataset_name, dataset_full_path, opt, metrics)
        if result:
            all_results.update(result)

    # 6. 汇总所有结果
    print("\n" + "="*50)
    print("所有数据集测试结果汇总:")
    print("-"*50)
    print(f"{'数据集名称':<15} | MAE     | Sm     ")
    print("-"*50)
    for dataset, res in all_results.items():
        print(f"{dataset:<15} | {res['MAE']:.4f} | {res['Sm']:.4f} ")
    print("="*50)



    print("\n所有数据集测试完成！结果已保存到TensorBoard日志")
"""

TagNet_tiny_RGBDT29K_100e.pth
==================================================
所有数据集测试结果汇总:
--------------------------------------------------
数据集名称           | MAE     | Sm
--------------------------------------------------
DUT-RGBD-Test   | 0.0234 | 0.9397
LFSD            | 0.0541 | 0.8849
NJUD            | 0.0349 | 0.9188
NLPR            | 0.0208 | 0.9314
SIP             | 0.0517 | 0.8836
SSD             | 0.0448 | 0.8710
STERE           | 0.0306 | 0.9221
==================================================


TagNet_small_RGBDT29K_100e.pth
==================================================
所有数据集测试结果汇总:
--------------------------------------------------
数据集名称           | MAE     | Sm
--------------------------------------------------
DUT-RGBD-Test   | 0.0241 | 0.9400
LFSD            | 0.0526 | 0.8808
NJUD            | 0.0328 | 0.9212
NLPR            | 0.0196 | 0.9333
SIP             | 0.0457 | 0.8926
SSD             | 0.0439 | 0.8728
STERE           | 0.0287 | 0.9267
==================================================



"""
