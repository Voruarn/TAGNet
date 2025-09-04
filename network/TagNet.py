import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import BertModel, BertTokenizer, VisionEncoderDecoderModel, ViTImageProcessor
import numpy as np

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeatureExtractor(nn.Module):
    """特征提取模块，用于提取RGB和深度图的特征"""
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # 使用ResNet50作为基础模型
        self.resnet = models.resnet50(pretrained=True)
        
        # 移除最后的全连接层，保留卷积特征提取部分
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # 深度特征提取分支（参数独立）
        self.depth_features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            *list(self.resnet.children())[4:-2]  # 复用ResNet的后续层
        )
        
        # 冻结部分底层参数
        for param in list(self.features.parameters())[:-10]:
            param.requires_grad = False
        
        for param in list(self.depth_features.parameters())[:-10]:
            param.requires_grad = False

    def forward(self, rgb, depth):
        """
        提取RGB和深度特征
        rgb: (B, 3, H, W)
        depth: (B, 1, H, W)
        """
        # RGB特征
        rgb_feat = self.features(rgb)  # (B, 2048, H/32, W/32)
        
        # 深度特征
        depth_feat = self.depth_features(depth)  # (B, 2048, H/32, W/32)
        
        # 提取不同尺度的特征
        # 这里简化处理，实际中可以从不同层提取多尺度特征
        return {
            'rgb': rgb_feat,
            'depth': depth_feat,
            'rgb_low': self.features[:6](rgb),  # 较早层的特征，保留更多细节
            'depth_low': self.depth_features[:6](depth)
        }

class VLMSalientTextGenerator(nn.Module):
    """VLM显著性文本生成模块"""
    def __init__(self):
        super(VLMSalientTextGenerator, self).__init__()
        # 使用预训练的视觉-语言模型，如ViT-GPT2
        self.vlm = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        # 提示词模板
        self.prompt_template = "Describe the salient object in the image."
        
        # 冻结VLM参数
        for param in self.vlm.parameters():
            param.requires_grad = False

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
            # 转换为PIL图像格式（VLM通常需要这种输入格式）
            rgb_img = (rgb[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            # 准备输入
            pixel_values = self.image_processor(images=rgb_img, return_tensors="pt").pixel_values.to(device)
            
            # 生成文本
            output_ids = self.vlm.generate(
                pixel_values,
                max_length=50,
                num_beams=4,
                repetition_penalty=1.5,
                length_penalty=1.0,
                early_stopping=True
            )
            
            # 解码生成的文本
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # 结合提示词
            full_text = self.prompt_template + generated_text
            texts.append(full_text)
        
        return texts

    def forward(self, rgb, depth):
        """生成显著性文本并进行编码"""
        # 生成文本
        texts = self.generate_salient_text(rgb, depth)
        
        # 文本编码
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=100
        ).to(device)
        
        return {
            "texts": texts,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }

class TextSemanticEncoder(nn.Module):
    """文本语义编码模块"""
    def __init__(self):
        super(TextSemanticEncoder, self).__init__()
        # 使用BERT作为文本编码器
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        # 冻结部分BERT参数
        for param in list(self.bert.parameters())[:-10]:
            param.requires_grad = False
        
        # 语义聚焦层，强化关键词对应的特征
        self.semantic_focus = nn.Sequential(
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Linear(768, 768)
        )

    def forward(self, input_ids, attention_mask):
        """
        编码文本语义
        input_ids: (B, T)
        attention_mask: (B, T)
        """
        # BERT编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_feat = outputs.last_hidden_state[:, 0, :]  # (B, 768)，取[CLS] token的特征
        
        # 语义聚焦
        focused_feat = self.semantic_focus(cls_feat)
        
        return focused_feat

class CrossModalFusion(nn.Module):
    """跨模态特征融合模块"""
    def __init__(self, visual_dim=2048, text_dim=768):
        super(CrossModalFusion, self).__init__()
        # 特征维度调整
        self.text_proj = nn.Linear(text_dim, visual_dim)
        self.rgb_proj = nn.Conv2d(visual_dim, visual_dim, kernel_size=1)
        self.depth_proj = nn.Conv2d(visual_dim, visual_dim, kernel_size=1)
        
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

    def forward(self, visual_feats, text_feat, rgb, depth):
        """
        融合视觉特征和文本特征
        visual_feats: 包含rgb和depth特征的字典
        text_feat: (B, text_dim)
        rgb: 原始RGB图像 (B, 3, H, W)
        depth: 原始深度图 (B, 1, H, W)
        """
        rgb_feat = visual_feats['rgb']  # (B, 2048, H', W')
        depth_feat = visual_feats['depth']  # (B, 2048, H', W')
        b, c, h, w = rgb_feat.shape
        
        # 文本特征投影到视觉特征维度
        text_proj = self.text_proj(text_feat)  # (B, 2048)
        
        # 调整RGB特征
        rgb_proj = self.rgb_proj(rgb_feat)  # (B, 2048, H', W')
        
        # 调整深度特征
        depth_proj = self.depth_proj(depth_feat)  # (B, 2048, H', W')
        
        # 为注意力机制准备特征
        rgb_flat = rgb_proj.flatten(2).permute(0, 2, 1)  # (B, H'*W', 2048)
        depth_flat = depth_proj.flatten(2).permute(0, 2, 1)  # (B, H'*W', 2048)
        text_expanded = text_proj.unsqueeze(1)  # (B, 1, 2048)
        
        # RGB-文本注意力融合
        rgb_text_attn, _ = self.rgb_text_attention(rgb_flat, text_expanded, text_expanded)
        rgb_text_feat = rgb_text_attn.permute(0, 2, 1).view(b, c, h, w)  # (B, 2048, H', W')
        
        # 深度-文本注意力融合
        depth_text_attn, _ = self.depth_text_attention(depth_flat, text_expanded, text_expanded)
        depth_text_feat = depth_text_attn.permute(0, 2, 1).view(b, c, h, w)  # (B, 2048, H', W')
        
        # 计算门控权重
        gate_input = torch.cat([rgb_text_feat, depth_text_feat], dim=1)
        gate_weight = self.gate(gate_input)  # (B, 2048, H', W')
        
        # 融合特征
        fused_feat = gate_weight * rgb_text_feat + (1 - gate_weight) * depth_text_feat
        
        # 计算语义-像素匹配损失（简化实现）
        # 这里仅作为示例，实际中需要更复杂的实现来关联文本语义和像素特征
        pixel_loss = self.semantic_pixel_loss(rgb_proj, depth_proj)
        
        # 计算深度语义一致性损失（简化实现）
        depth_loss = self.depth_semantic_loss(depth_proj, rgb_proj.detach())
        
        return {
            "fused_feat": fused_feat,
            "losses": {
                "pixel_loss": pixel_loss,
                "depth_loss": depth_loss
            }
        }

class MultiScaleOptimizer(nn.Module):
    """多尺度特征优化模块"""
    def __init__(self, in_channels=2048):
        super(MultiScaleOptimizer, self).__init__()
        # 不同尺度的特征处理
        self.scale1 = nn.Sequential(
            nn.Conv2d(in_channels, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        
        self.scale2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        self.scale3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # 跨尺度注意力
        self.cross_attention = nn.MultiheadAttention(256, num_heads=8, batch_first=True)
        
        # 上采样模块
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, fused_feat, low_level_feats):
        """
        优化多尺度特征
        fused_feat: 融合后的特征 (B, 2048, H', W')
        low_level_feats: 包含低级特征的字典
        """
        # 处理不同尺度
        feat1 = self.scale1(fused_feat)  # (B, 1024, H', W')
        feat2 = self.scale2(self.upsample(feat1))  # (B, 512, 2H', 2W')
        feat3 = self.scale3(self.upsample(feat2))  # (B, 256, 4H', 4W')
        
        # 获取低级特征并调整维度
        rgb_low = F.interpolate(
            low_level_feats['rgb_low'], 
            size=feat3.shape[2:], 
            mode='bilinear', 
            align_corners=True
        )
        depth_low = F.interpolate(
            low_level_feats['depth_low'], 
            size=feat3.shape[2:], 
            mode='bilinear', 
            align_corners=True
        )
        
        # 调整低级特征通道数
        rgb_low = nn.Conv2d(rgb_low.size(1), 256, kernel_size=1).to(device)(rgb_low)
        depth_low = nn.Conv2d(depth_low.size(1), 256, kernel_size=1).to(device)(depth_low)
        
        # 跨尺度注意力融合
        b, c, h, w = feat3.shape
        feat3_flat = feat3.flatten(2).permute(0, 2, 1)  # (B, H*W, 256)
        rgb_low_flat = rgb_low.flatten(2).permute(0, 2, 1)  # (B, H*W, 256)
        
        attn_output, _ = self.cross_attention(feat3_flat, rgb_low_flat, depth_low.flatten(2).permute(0, 2, 1))
        attn_feat = attn_output.permute(0, 2, 1).view(b, c, h, w)  # (B, 256, H, W)
        
        # 多尺度特征融合
        final_feat = feat3 + attn_feat + rgb_low + depth_low
        
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

class VLMRGBDSOD(nn.Module):
    """VLM引导的RGB-D显著性目标检测模型"""
    def __init__(self):
        super(VLMRGBDSOD, self).__init__()
        # 各组件模块
        self.feature_extractor = FeatureExtractor()
        self.vlm_text_generator = VLMSalientTextGenerator()
        self.text_encoder = TextSemanticEncoder()
        self.cross_modal_fusion = CrossModalFusion()
        self.multi_scale_optimizer = MultiScaleOptimizer()
        self.salient_predictor = SalientPredictor()
        
        # 损失函数
        self.bce_loss = nn.BCELoss()
        self.iou_loss = lambda x, y: 1 - torch.mean((x * y).sum(dim=[1,2,3]) / 
                                                    (x + y - x * y).sum(dim=[1,2,3]) + 1e-8)
        self.semantic_consistency_loss = nn.CosineEmbeddingLoss()

    def forward(self, rgb, depth, gt=None):
        """
        前向传播
        rgb: (B, 3, H, W)
        depth: (B, 1, H, W)
        gt: Ground Truth显著性图 (B, 1, H, W)，训练时提供
        """
        # 1. 特征提取
        visual_feats = self.feature_extractor(rgb, depth)
        
        # 2. 生成显著性文本
        # 在训练时可以减少生成频率以提高效率
        if self.training and torch.rand(1) < 0.2:  # 20%概率生成新文本
            text_outputs = self.vlm_text_generator(rgb, depth)
            # 缓存文本特征
            self.cached_text_feat = self.text_encoder(
                text_outputs["input_ids"], 
                text_outputs["attention_mask"]
            )
        elif self.training:  # 使用缓存的文本特征
            text_feat = self.cached_text_feat
        else:  # 推理时总是生成新文本
            text_outputs = self.vlm_text_generator(rgb, depth)
            text_feat = self.text_encoder(
                text_outputs["input_ids"], 
                text_outputs["attention_mask"]
            )
        
        # 3. 跨模态特征融合
        fusion_output = self.cross_modal_fusion(visual_feats, text_feat, rgb, depth)
        
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
            # 将预测图特征与文本特征进行对比
            pred_feat = F.adaptive_avg_pool2d(optimized_feat, (1, 1)).squeeze()  # (B, 256)
            text_proj = nn.Linear(768, 256).to(device)(text_feat)  # 投影到相同维度
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
            "texts": text_outputs["texts"] if not self.training else None
        }

# 训练示例
def train_example():
    # 创建模型
    model = VLMRGBDSOD().to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # 模拟数据
    batch_size = 2
    H, W = 256, 256
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
    model = VLMRGBDSOD().to(device)
    model.eval()
    
    # 模拟数据
    batch_size = 1
    H, W = 256, 256
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
