import torch
from network.TAGNet import TAGNet

def test_model():
    torch.manual_seed(42)
    model = TAGNet(convnext_model_name='convnext_base')
    
    batch_size = 2
    height, width = 256, 256  
    rgb = torch.randn(batch_size, 3, height, width)  # RGB图像 (B, 3, H, W)
    depth = torch.randn(batch_size, 1, height, width)  # 深度图 (B, 1, H, W)
    target = torch.randint(0, 1, (batch_size, 1, height, width), dtype=torch.float32)  # 目标显著性图

    texts = [
        "A salient object in the center of the image with clear edges",
        "A small object on the left side, distinct from the background"
    ]
    
    model.train()

    print("=== 训练模式测试 ===")
    print(f"输入RGB形状: {rgb.shape}")
    print(f"输入深度形状: {depth.shape}")
    print(f"输入文本数量: {len(texts)}")

    if torch.cuda.is_available():
        model = model.cuda()
        rgb_cuda = rgb.cuda()
        depth_cuda = depth.cuda()
        target_cuda = target.cuda()

        model.train()
        outputs_cuda = model(rgb_cuda, depth_cuda, texts, target_cuda)
        print("\n=== CUDA训练模式测试 ===")
        print(f"CUDA总损失: {outputs_cuda['losses']['total_loss'].item():.6f}")
        
        model.eval()
        with torch.no_grad():
            outputs_cuda = model(rgb_cuda, depth_cuda, texts)
        print("=== CUDA推理模式测试 ===")
        print(f"CUDA推理输出形状: {outputs_cuda['sal_map'].shape}")

if __name__ == "__main__":
    print("TAGNet Test...")
    test_model()
    print("Test Done !")
