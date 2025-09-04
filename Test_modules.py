import torch
# 假设FastVLMDinoV3CLIPSOD模型定义在该路径下
from network.TagNetV3 import TagNet


# 测试代码
def test_model():
    # 设置随机种子
    torch.manual_seed(42)
    
    # 创建模型（可指定ConvNext型号，如convnext_small/convnext_base）
    model = TagNet(convnext_model_name='convnext_small')
    
    # 创建随机输入数据
    batch_size = 2
    height, width = 256, 256  # 模型默认输入尺寸
    rgb = torch.randn(batch_size, 3, height, width)  # RGB图像 (B, 3, H, W)
    depth = torch.randn(batch_size, 1, height, width)  # 深度图 (B, 1, H, W)
    target = torch.randint(0, 2, (batch_size, 1, height, width), dtype=torch.float32)  # 目标显著性图
    
    # 模拟文本描述（与图像对应的显著性文本，实际应用中应从数据集加载）
    texts = [
        "A salient object in the center of the image with clear edges",
        "A small object on the left side, distinct from the background"
    ]
    
    # 训练模式下的前向传播（带损失计算）
    model.train()

    # 打印训练模式结果
    print("=== 训练模式测试 ===")
    print(f"输入RGB形状: {rgb.shape}")
    print(f"输入深度形状: {depth.shape}")
    print(f"输入文本数量: {len(texts)}")

    # 测试CUDA支持
    if torch.cuda.is_available():
        model = model.cuda()
        rgb_cuda = rgb.cuda()
        depth_cuda = depth.cuda()
        target_cuda = target.cuda()
        
        # 训练模式CUDA测试
        model.train()
        outputs_cuda = model(rgb_cuda, depth_cuda, texts, target_cuda)
        print("\n=== CUDA训练模式测试 ===")
        print(f"CUDA总损失: {outputs_cuda['losses']['total_loss'].item():.6f}")
        
        # 推理模式CUDA测试
        model.eval()
        with torch.no_grad():
            outputs_cuda = model(rgb_cuda, depth_cuda, texts)
        print("=== CUDA推理模式测试 ===")
        print(f"CUDA推理输出形状: {outputs_cuda['sal_map'].shape}")


if __name__ == "__main__":
    print("FastVLMDinoV3CLIPSOD模型测试开始...")
    test_model()
    print("测试完成!")
