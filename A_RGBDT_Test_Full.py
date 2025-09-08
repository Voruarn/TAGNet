import torch
import numpy as np
import random
import os, argparse
from tqdm import tqdm
from setting.VLdataLoader import get_loader
from network.TagNetV3 import TagNet
from metrics.SOD_metrics_Full import SODMetrics  # 导入优化后的指标计算类
import time



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
parser.add_argument('--convnext_model', type=str, default='convnext_tiny', 
                    help='ConvNext backbone: [convnext_tiny, convnext_small, convnext_base]')
parser.add_argument("--load", type=str, default="/fuyuxiang/Projects/CLS/Seg/TagNet/CHKP0905/TagNet_tiny_RGBDT29K_100e.pth", help="模型权重文件路径")
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
    """验证单个数据集，返回所有指标结果"""
    metrics.reset()
    with torch.no_grad():
        for step, (images, depths, texts, labels) in tqdm(enumerate(loader), desc="测试中", leave=False):
            # 数据设备迁移
            images = images.to(device, dtype=torch.float32)
            depths = depths.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            # 模型推理
            outputs = model(images, depths, texts)
            preds = outputs['sal_map']  # 形状应为 (B, 1, H, W)
            
            # 更新指标计算器
            metrics.update(preds, labels)

    return metrics.get_results()


def test_on_dataset(model, dataset_name, data_path, opt, metrics, log_file):
    """测试单个数据集并返回结果，同时写入日志文件"""
    print(f"\n===== 开始测试数据集: {dataset_name} =====")

    # 构建数据集路径
    img_path = os.path.join(data_path, 'test_images/')
    depth_path = os.path.join(data_path, 'gen_depth/')
    mask_path = os.path.join(data_path, 'test_masks/')
    text_path = os.path.join(data_path, 'test_text_05b/')
    
    # 检查路径是否存在
    for path in [img_path, depth_path, mask_path, text_path]:
        if not os.path.exists(path):
            warn_msg = f"警告: 路径不存在 - {path}，跳过该数据集"
            print(warn_msg)
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
    data_info = f"加载数据集 {dataset_name}，共 {test_num} 张图像"
    print(data_info)

    # 执行测试
    model.eval()
    results = validate(model, test_loader, device, metrics)
    
    # 打印并记录当前数据集的完整指标结果
    result_strs = [
        f"\n数据集 {dataset_name} 测试结果:",
        f"MAE:         {results['MAE']:.4f}",
        f"Smeasure:    {results['Smeasure']:.4f}",
        f"Fmeasure:    max={results['Fmeasure']['max']:.4f}, mean={results['Fmeasure']['mean']:.4f}",
        f"Emeasure:    max={results['Emeasure']['max']:.4f}, mean={results['Emeasure']['mean']:.4f}",
        "-" * 60
    ]
    
    for s in result_strs:
        print(s)
    
    return {dataset_name: results}


if __name__=='__main__':
    # 创建日志目录
    os.makedirs(opt.log_dir, exist_ok=True)
    # 生成带时间戳的日志文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(opt.log_dir, f"test_results_{timestamp}.txt")
    
    # 记录配置信息
    with open(log_file, 'a') as f:
        f.write("="*80 + "\n")
        f.write("测试配置信息:\n")
        f.write(f"模型名称: {opt.model}\n")
        f.write(f"Backbone: {opt.convnext_model}\n")
        f.write(f"权重路径: {opt.load}\n")
        f.write(f"测试图像尺寸: {opt.testsize}\n")
        f.write(f"批次大小: {opt.batchsize}\n")
        f.write(f"设备: {device}\n")
        f.write("="*80 + "\n\n")

    print(f"[配置信息] {opt}")
    print(f"[日志保存] 结果将保存至: {log_file}")

    # 初始化模型
    model = TagNet(convnext_model_name=opt.convnext_model).to(device)
    print(f"[模型加载] 加载TagNet，backbone: {opt.convnext_model}")

    # 加载预训练权重
    if os.path.isfile(opt.load):
        checkpoint = torch.load(opt.load, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"[权重加载] 从 {opt.load} 恢复模型权重")
        with open(log_file, 'a') as f:
            f.write(f"[权重加载] 从 {opt.load} 恢复模型权重\n")
    else:
        error_msg = f"[错误] 权重文件不存在: {opt.load}"
        print(error_msg)
        with open(log_file, 'a') as f:
            f.write(error_msg + "\n")
        exit(1)

    # 初始化评估指标计算器
    metrics = SODMetrics(cuda=device.type == 'cuda')

    # 批量测试所有数据集
    all_results = {}
    for dataset_name, subpath in DATASETS.items():
        dataset_full_path = os.path.join(opt.base_data_path, subpath)
        result = test_on_dataset(model, dataset_name, dataset_full_path, opt, metrics, log_file)
        if result:
            all_results.update(result)

    # 汇总所有结果并格式化输出
    summary_strs = ["\n" + "="*80, "所有数据集测试结果汇总:", "-"*80]
    # 打印表头
    header = (f"{'数据集名称':<15} | {'MAE':<6} | {'Sm':<6} | "
              f"Fmax  | Fmean | Emax  | Emean  ")
    summary_strs.append(header)
    summary_strs.append("-"*80)
    
    # 打印每个数据集的结果
    for dataset, res in all_results.items():
        line = (f"{dataset:<15} | "
                f"{res['MAE']:.4f} | "
                f"{res['Smeasure']:.4f} | "
                f"{res['Fmeasure']['max']:.4f} | "
                f"{res['Fmeasure']['mean']:.4f} | "
                f"{res['Emeasure']['max']:.4f} | "
                f"{res['Emeasure']['mean']:.4f}")
        summary_strs.append(line)
    summary_strs.append("="*80)

    # 打印并保存汇总结果
    for s in summary_strs:
        print(s)
        with open(log_file, 'a') as f:
            f.write(s + "\n")

    print(f"\n所有数据集测试完成！结果已保存至: {log_file}")

"""


"""
