import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer, AutoProcessor
import clip
from PIL import Image
import numpy as np
from convnext import convnext_tiny, convnext_small, convnext_base

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DinoV3ConvNextFeatureExtractor(nn.Module):
    """使用DinoV3预训练的ConvNext模型提取RGB和深度特征"""
    embed_dims = {
        "convnext_tiny":[96, 192, 384, 768],
        "convnext_small":[96, 192, 384, 768],
        "convnext_base":[128, 256, 512, 1024]
    }
    def __init__(self, model_name='convnext_small', pretrained=True):
        super(DinoV3ConvNextFeatureExtractor, self).__init__()
        # 加载DinoV3预训练的ConvNext模型
        self.convnext = eval(model_name)(pretrained=pretrained)

        # 冻结所有参数
        for param in self.convnext.parameters():
            param.requires_grad = False
        
        # 特征维度 (ConvNextV2-Base输出维度为1024)
        self.feature_dim = self.embed_dims[model_name][-1]
        
        # 用于处理深度图的输入适配层
        # 将单通道深度图转换为3通道，以便输入ConvNext
        self.depth_adapter = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)
        # 这个适配层的参数是可训练的
        # 其他所有参数都冻结

    def forward(self, rgb, depth):
        """
        提取RGB和深度特征
        rgb: (B, 3, H, W)
        depth: (B, 1, H, W)
        """
        # 处理深度图 - 转换为3通道
        depth = self.depth_adapter(depth)  # (B, 3, H, W)
        
        # 提取RGB特征
        rgb_outputs = self.convnext(rgb)
        # 获取不同阶段的特征
        rgb_features = {
            'c2': rgb_outputs[1],  # 早期特征，高分辨率，低维度
            'c3': rgb_outputs[2],
            'c4': rgb_outputs[3]   # 晚期特征，低分辨率，高维度
        }
        
        # 提取深度特征
        depth_outputs = self.convnext(depth)
        depth_features = {
            'c2': depth_outputs[1],
            'c3': depth_outputs[2],
            'c4': depth_outputs[3]
        }
        
        return {
            'rgb': rgb_features,
            'depth': depth_features
        }

class FastVLMSalientTextGenerator(nn.Module):
    """FastVLM显著性文本生成模块"""
    def __init__(self):
        super(FastVLMSalientTextGenerator, self).__init__()
        # 加载FastVLM模型和处理器
        self.model = AutoModel.from_pretrained("BAAI/FastVLM-7B", trust_remote_code=True).to(device)
        self.processor = AutoProcessor.from_pretrained("BAAI/FastVLM-7B", trust_remote_code=True)
        
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 显著性描述提示词
        self.prompt = "Describe the salient object in the image."
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def generate_salient_text(self, rgb, depth):
        """
        生成显著性文本描述
        rgb: (B, 3, H, W)
        depth: (B, 1, H, W)
        """
        batch_size = rgb.size(0)
        texts = []
        
        # 对批次中的每个样本生成文本
        for i in range(batch_size):
            # 处理RGB图像
            rgb_img = rgb[i].cpu()
            rgb_img = self.transform(rgb_img)  # 应用标准化
            rgb_img = transforms.ToPILImage()(rgb_img)
            
            # 处理深度图，转换为伪彩色图以便FastVLM理解
            depth_np = depth[i, 0].cpu().numpy()
            depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
            depth_np = (depth_np * 255).astype(np.uint8)
            depth_img = Image.fromarray(depth_np).convert('L').convert('RGB')  # 转为3通道
            
            # 将RGB和深度图拼接在一起，让模型同时看到
            combined_img = Image.new('RGB', (448, 224))
            combined_img.paste(rgb_img, (0, 0))
            combined_img.paste(depth_img, (224, 0))
            
            # 准备输入
            inputs = self.processor(images=combined_img, text=self.prompt, return_tensors="pt").to(device)
            
            # 生成文本
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=3,
                repetition_penalty=1.5,
                length_penalty=1.0,
                early_stopping=True
            )
            
            # 解码生成的文本
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # 提取模型对提示词的响应部分
            if self.prompt in generated_text:
                generated_text = generated_text.replace(self.prompt, "").strip()
            
            texts.append(generated_text)
        
        return texts

    def forward(self, rgb, depth):
        """生成显著性文本"""
        # 生成文本
        texts = self.generate_salient_text(rgb, depth)
        return {"texts": texts}

class CLIPTextEncoder(nn.Module):
    """使用CLIP的文本编码器"""
    def __init__(self):
        super(CLIPTextEncoder, self).__init__()
        # 加载CLIP模型
        self.clip_model, _ = clip.load("ViT-B/16", device=device)
        
        # 冻结所有参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # CLIP文本编码器输出维度
        self.output_dim = 768

    def forward(self, texts):
        """
        编码文本语义
        texts: 字符串列表 (B,)
        """
        # 使用CLIP的tokenizer处理文本
        text_tokens = clip.tokenize(texts).to(device)
        
        # 编码文本
        with torch.no_grad():  # 确保不计算梯度
            text_features = self.clip_model.encode_text(text_tokens)
        
        # 归一化特征
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features

class CrossModalFusion(nn.Module):
    """跨模态特征融合模块"""
    def __init__(self, visual_dim=1024, text_dim=512):
        super(CrossModalFusion, self).__init__()
        # 文本特征维度映射到视觉特征维度
        self.text_proj = nn.Linear(text_dim, visual_dim)
        
        # RGB-文本注意力融合
        self.rgb_text_attention = nn.MultiheadAttention(visual_dim, num_heads=8, batch_first=True)
        
        # 深度-文本注意力融合
        self.depth_text_attention = nn.MultiheadAttention(visual_dim, num_heads=8, batch_first=True)
        
        # 门控单元，平衡两个路径的特征
        self.gate = nn.Sequential(
            nn.Conv2d(visual_dim * 2, visual_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 语义-像素匹配损失
        self.semantic_pixel_loss = nn.MSELoss()
        
        # 深度语义一致性损失
        self.depth_semantic_loss = nn.L1Loss()

    def forward(self, visual_feats, text_feat):
        """
        融合视觉特征和文本特征
        visual_feats: 包含rgb和depth特征的字典
        text_feat: (B, text_dim)
        """
        # 获取高层特征用于融合
        rgb_feat = visual_feats['rgb']['c4']  # (B, 1024, H', W')
        depth_feat = visual_feats['depth']['c4']  # (B, 1024, H', W')
        b, c, h, w = rgb_feat.shape
        
        # 文本特征投影到视觉特征维度
        text_proj = self.text_proj(text_feat)  # (B, 1024)
        
        # 为注意力机制准备特征
        rgb_flat = rgb_feat.flatten(2).permute(0, 2, 1)  # (B, H'*W', 1024)
        depth_flat = depth_feat.flatten(2).permute(0, 2, 1)  # (B, H'*W', 1024)
        text_expanded = text_proj.unsqueeze(1)  # (B, 1, 1024)
        
        # RGB-文本注意力融合
        rgb_text_attn, _ = self.rgb_text_attention(rgb_flat, text_expanded, text_expanded)
        rgb_text_feat = rgb_text_attn.permute(0, 2, 1).view(b, c, h, w)  # (B, 1024, H', W')
        
        # 深度-文本注意力融合
        depth_text_attn, _ = self.depth_text_attention(depth_flat, text_expanded, text_expanded)
        depth_text_feat = depth_text_attn.permute(0, 2, 1).view(b, c, h, w)  # (B, 1024, H', W')
        
        # 计算门控权重
        gate_input = torch.cat([rgb_text_feat, depth_text_feat], dim=1)
        gate_weight = self.gate(gate_input)  # (B, 1024, H', W')
        
        # 融合特征
        fused_feat = gate_weight * rgb_text_feat + (1 - gate_weight) * depth_text_feat
        
        # 计算语义-像素匹配损失
        pixel_loss = self.semantic_pixel_loss(rgb_feat, depth_feat)
        
        # 计算深度语义一致性损失
        depth_loss = self.depth_semantic_loss(depth_feat, rgb_feat.detach())
        
        return {
            "fused_feat": fused_feat,
            "losses": {
                "pixel_loss": pixel_loss,
                "depth_loss": depth_loss
            }
        }

class MultiScaleOptimizer(nn.Module):
    """多尺度特征优化模块"""
    def __init__(self, in_channels=1024):
        super(MultiScaleOptimizer, self).__init__()
        # 调整不同尺度特征的通道数
        self.scale_c2 = nn.Conv2d(256, 256, kernel_size=1)
        self.scale_c3 = nn.Conv2d(512, 256, kernel_size=1)
        self.scale_c4 = nn.Conv2d(1024, 256, kernel_size=1)
        
        # 跨尺度注意力
        self.cross_attention = nn.MultiheadAttention(256, num_heads=8, batch_first=True)
        
        # 融合后的特征处理
        self.fusion_processor = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def forward(self, fused_feat, multi_scale_feats):
        """
        优化多尺度特征
        fused_feat: 融合后的特征 (B, 1024, H', W')
        multi_scale_feats: 包含不同尺度特征的字典
        """
        # 获取各尺度特征
        rgb_c2 = multi_scale_feats['rgb']['c2']
        rgb_c3 = multi_scale_feats['rgb']['c3']
        depth_c2 = multi_scale_feats['depth']['c2']
        depth_c3 = multi_scale_feats['depth']['c3']
        
        # 调整通道数
        rgb_c2 = self.scale_c2(rgb_c2)
        rgb_c3 = self.scale_c3(rgb_c3)
        depth_c2 = self.scale_c2(depth_c2)
        depth_c3 = self.scale_c3(depth_c3)
        fused_c4 = self.scale_c4(fused_feat)
        
        # 调整到相同空间尺寸（使用最大的尺寸）
        target_h, target_w = rgb_c2.shape[2], rgb_c2.shape[3]
        rgb_c3_up = F.interpolate(rgb_c3, size=(target_h, target_w), mode='bilinear', align_corners=True)
        depth_c3_up = F.interpolate(depth_c3, size=(target_h, target_w), mode='bilinear', align_corners=True)
        fused_c4_up = F.interpolate(fused_c4, size=(target_h, target_w), mode='bilinear', align_corners=True)
        
        # 跨尺度注意力融合
        b, c, h, w = rgb_c2.shape
        feat_c2_flat = rgb_c2.flatten(2).permute(0, 2, 1)  # (B, H*W, 256)
        feat_c3_flat = rgb_c3_up.flatten(2).permute(0, 2, 1)  # (B, H*W, 256)
        feat_c4_flat = fused_c4_up.flatten(2).permute(0, 2, 1)  # (B, H*W, 256)
        
        # 注意力融合
        attn_output, _ = self.cross_attention(feat_c4_flat, feat_c3_flat, feat_c2_flat)
        attn_feat = attn_output.permute(0, 2, 1).view(b, c, h, w)  # (B, 256, H, W)
        
        # 多尺度特征融合
        final_feat = attn_feat + rgb_c2 + depth_c2 + rgb_c3_up + depth_c3_up + fused_c4_up
        
        # 进一步处理融合特征
        final_feat = self.fusion_processor(final_feat)
        
        return final_feat

class SalientPredictor(nn.Module):
    """显著性预测头"""
    def __init__(self, in_channels=256):
        super(SalientPredictor, self).__init__()
        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()  # 输出0-1之间的显著性分数
        )

    def forward(self, feat, target_size):
        """
        预测显著性图
        feat: 优化后的特征 (B, 256, H, W)
        target_size: 目标输出尺寸 (H_out, W_out)
        """
        pred = self.predictor(feat)
        # 调整到目标尺寸
        pred = F.interpolate(pred, size=target_size, mode='bilinear', align_corners=True)
        return pred

class FastVLMDinoV3CLIPSOD(nn.Module):
    """FastVLM-DinoV3-CLIP引导的RGB-D显著性目标检测模型"""
    def __init__(self):
        super(FastVLMDinoV3CLIPSOD, self).__init__()
        # 各组件模块
        self.feature_extractor = DinoV3ConvNextFeatureExtractor()
        # self.vlm_text_generator = FastVLMSalientTextGenerator()
        self.text_encoder = CLIPTextEncoder()
        self.cross_modal_fusion = CrossModalFusion()
        self.multi_scale_optimizer = MultiScaleOptimizer()
        self.salient_predictor = SalientPredictor()
        
        # 损失函数
        self.bce_loss = nn.BCELoss()
        self.iou_loss = lambda x, y: 1 - torch.mean((x * y).sum(dim=[1,2,3]) / 
                                                    (x + y - x * y).sum(dim=[1,2,3]) + 1e-8)
        self.semantic_consistency_loss = nn.CosineEmbeddingLoss()
        
        # 用于缓存文本特征，减少训练时的计算开销
        self.cached_text_feats = None

    def forward(self, rgb, depth, text, gt=None):
        """
        前向传播
        rgb: (B, 3, H, W)
        depth: (B, 1, H, W)
        gt: Ground Truth显著性图 (B, 1, H, W)，训练时提供
        """
        # 1. 特征提取
        visual_feats = self.feature_extractor(rgb, depth)
        
        # 2. 基于预先VLM生成的显著性文本进行编码
        text_feat = self.text_encoder(text)

        # 3. 跨模态特征融合
        fusion_output = self.cross_modal_fusion(visual_feats, text_feat)
        
        # 4. 多尺度特征优化
        optimized_feat = self.multi_scale_optimizer(
            fusion_output["fused_feat"], 
            visual_feats
        )
        
        # 5. 显著性预测
        pred = self.salient_predictor(optimized_feat, rgb.shape[2:])
        
        # 计算损失（训练阶段）
        losses = {}
        if self.training and gt is not None:
            # BCE+IoU损失
            bce_loss = self.bce_loss(pred, gt)
            iou_loss = self.iou_loss(pred, gt)
            losses["bce_iou_loss"] = bce_loss + iou_loss
            
            # 语义一致性损失
            pred_feat = F.adaptive_avg_pool2d(optimized_feat, (1, 1)).squeeze()  # (B, 256)
            text_proj = nn.Linear(512, 256).to(device)(text_feat)  # 投影到相同维度
            semantic_loss = self.semantic_consistency_loss(
                pred_feat, text_proj, torch.ones(pred_feat.size(0)).to(device)
            )
            losses["semantic_loss"] = semantic_loss
            
            # 来自融合模块的损失
            losses.update(fusion_output["losses"])
            
            # 总损失
            total_loss = (
                losses["bce_iou_loss"] + 
                0.3 * losses["semantic_loss"] + 
                0.2 * losses["pixel_loss"] + 
                0.2 * losses["depth_loss"]
            )
            losses["total_loss"] = total_loss
        
        return {
            "pred": pred,
            "losses": losses if self.training else None,
        }

# 训练示例
def train_example():
    # 创建模型
    model = FastVLMDinoV3CLIPSOD().to(device)
    
    # 只优化需要训练的参数
    params_to_train = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_train, lr=1e-4, weight_decay=1e-5)
    
    # 模拟数据
    batch_size = 2
    H, W = 224, 224
    rgb = torch.randn(batch_size, 3, H, W).to(device)
    depth = torch.randn(batch_size, 1, H, W).to(device)
    gt = torch.rand(batch_size, 1, H, W).to(device)  # 模拟Ground Truth
    
    # 训练模式
    model.train()
    
    # 前向传播
    outputs = model(rgb, depth, gt)
    
    # 反向传播
    optimizer.zero_grad()
    outputs["losses"]["total_loss"].backward()
    optimizer.step()
    
    print("训练损失:", {k: v.item() for k, v in outputs["losses"].items()})

# 测试示例
def test_example():
    # 创建模型
    model = FastVLMDinoV3CLIPSOD().to(device)
    model.eval()
    
    # 模拟数据
    batch_size = 1
    H, W = 224, 224
    rgb = torch.randn(batch_size, 3, H, W).to(device)
    depth = torch.randn(batch_size, 1, H, W).to(device)
    
    # 推理
    with torch.no_grad():
        outputs = model(rgb, depth)
    
    print("生成的显著性文本:", outputs["texts"])
    print("预测的显著性图形状:", outputs["pred"].shape)

if __name__ == "__main__":
    # 运行训练示例
    train_example()
    
    # 运行测试示例
    test_example()
    
