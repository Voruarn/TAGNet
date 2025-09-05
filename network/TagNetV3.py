import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoModel, AutoProcessor
import clip
from PIL import Image
import numpy as np
# 假设convnext.py中定义了convnext_tiny/small/base，且返回各层特征列表（[c1, c2, c3, c4]）
from network.convnext import convnext_tiny, convnext_small, convnext_base

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DinoV3ConvNext(nn.Module):
    """
    DinoV3预训练ConvNext特征提取器（支持多型号）
    输出：RGB和深度的c2/c3/c4特征（对应ConvNext的第2/3/4阶段输出）
    """
    # 不同ConvNext型号的各阶段输出维度（与convnext.py中模型输出一致）
    embed_dims = {
        "convnext_tiny": [96, 192, 384, 768],    # c1, c2, c3, c4
        "convnext_small": [96, 192, 384, 768],
        "convnext_base": [128, 256, 512, 1024]
    }

    def __init__(self, model_name='convnext_tiny', pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.cur_embed_dims = self.embed_dims[model_name]  # 当前模型的维度配置
        
        # 加载DinoV3预训练ConvNext（确保convnext.py中模型forward返回[c1,c2,c3,c4]）
        self.convnext = eval(model_name)(pretrained=pretrained)
        
        # 冻结ConvNext所有参数（仅训练depth_adapter）
        # for param in self.convnext.parameters():
        #     param.requires_grad = False
        
        # 深度图适配层：单通道→3通道（匹配ConvNext输入），可训练
        self.depth_adapter = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)
        # 初始化适配层权重（提升深度特征初始化质量）
        nn.init.kaiming_normal_(self.depth_adapter.weight, mode='fan_out', nonlinearity='relu')
        if self.depth_adapter.bias is not None:
            nn.init.constant_(self.depth_adapter.bias, 0)

    def forward(self, rgb, depth):
        """
        Args:
            rgb: (B, 3, H, W) - RGB图像
            depth: (B, 1, H, W) - 单通道深度图
        Returns:
            feats: dict - 包含rgb和depth的c2/c3/c4特征
        """
        # 深度图适配：1ch→3ch（与RGB输入格式一致）
        depth_3ch = self.depth_adapter(depth)  # (B, 3, H, W)
        
        # 提取RGB特征（convnext返回[c1, c2, c3, c4]）
        rgb_c1, rgb_c2, rgb_c3, rgb_c4 = self.convnext(rgb)
        # 提取深度特征
        depth_c1, depth_c2, depth_c3, depth_c4 = self.convnext(depth_3ch)
        
        # 整理输出（仅保留c2/c3/c4，c1分辨率过高且语义弱）
        return {
            'rgb': {'c2': rgb_c2, 'c3': rgb_c3, 'c4': rgb_c4},
            'depth': {'c2': depth_c2, 'c3': depth_c3, 'c4': depth_c4}
        }


class CLIPTextEncoder(nn.Module):
    """
    CLIP文本编码器（ViT-B/16）- 冻结参数
    输出：768维归一化文本特征
    """
    def __init__(self):
        super().__init__()
        # 加载CLIP ViT-B/16（输出文本特征维度768）
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/16", device=device)
        
        # 冻结所有参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self.output_dim = 512  # CLIP ViT-B/16文本特征维度

    def forward(self, texts):
        """
        Args:
            texts: list[str] - 显著性文本列表（长度=batch_size）
        Returns:
            text_feats: (B, 768) - 归一化文本特征
        """
        # CLIP Tokenize（自动处理padding和截断）
        text_tokens = clip.tokenize(texts, truncate=True).to(device)
        
        # 编码（无梯度计算）
        with torch.no_grad():
            text_feats = self.clip_model.encode_text(text_tokens)
        
        # 特征归一化（提升跨模态匹配稳定性）
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        return text_feats


class CrossModalFusion(nn.Module):
    """
    跨模态融合模块（RGB-文本 + 深度-文本双路径）
    适配动态视觉/文本维度，仅训练融合相关参数
    """
    def __init__(self, visual_dim, text_dim):
        super().__init__()
        self.visual_dim = visual_dim  # 视觉特征维度（如convnext_small的c4=768）
        self.text_dim = text_dim      # 文本特征维度（CLIP=768）
        
        # 1. 文本特征→视觉特征维度映射（768→768/1024）
        self.text_proj = nn.Linear(text_dim, visual_dim)
        nn.init.kaiming_normal_(self.text_proj.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.text_proj.bias, 0)
        
        # 2. 注意力融合层（多头注意力，提升语义-视觉对齐）
        self.rgb_text_attn = nn.MultiheadAttention(visual_dim, num_heads=8, batch_first=True)
        self.depth_text_attn = nn.MultiheadAttention(visual_dim, num_heads=8, batch_first=True)
        
        # 3. 门控单元（动态平衡双路径特征）
        self.gate = nn.Sequential(
            nn.Conv2d(visual_dim * 2, visual_dim, kernel_size=1),
            nn.BatchNorm2d(visual_dim),  # 新增BN提升训练稳定性
            nn.Sigmoid()
        )
        # 门控单元初始化
        for m in self.gate.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 损失函数（复用但明确物理意义）
        self.semantic_pixel_loss = nn.MSELoss()  # 语义-像素匹配损失
        self.depth_semantic_loss = nn.L1Loss()   # 深度-语义一致性损失

    def forward(self, visual_feats, text_feat):
        """
        Args:
            visual_feats: dict - 包含rgb['c4']和depth['c4']的高层特征
            text_feat: (B, text_dim) - CLIP文本特征
        Returns:
            fused_feat: (B, visual_dim, H, W) - 融合后特征
            losses: dict - 融合相关损失
        """
        # 1. 提取高层视觉特征（c4，语义最强）
        rgb_c4 = visual_feats['rgb']['c4']  # (B, C, H, W)
        depth_c4 = visual_feats['depth']['c4']
        B, C, H, W = rgb_c4.shape
        
        # 2. 文本特征映射与扩展
        text_proj = self.text_proj(text_feat)  # (B, C)
        text_expand = text_proj.unsqueeze(1)   # (B, 1, C) - 适配注意力输入
        
        # 3. RGB-文本注意力融合
        rgb_flat = rgb_c4.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        rgb_text_attn, _ = self.rgb_text_attn(rgb_flat, text_expand, text_expand)
        rgb_text_feat = rgb_text_attn.permute(0, 2, 1).view(B, C, H, W)  # 恢复空间维度
        
        # 4. 深度-文本注意力融合（同RGB路径）
        depth_flat = depth_c4.flatten(2).permute(0, 2, 1)
        depth_text_attn, _ = self.depth_text_attn(depth_flat, text_expand, text_expand)
        depth_text_feat = depth_text_attn.permute(0, 2, 1).view(B, C, H, W)
        
        # 5. 门控融合（动态权重）
        gate_input = torch.cat([rgb_text_feat, depth_text_feat], dim=1)  # (B, 2C, H, W)
        gate_weight = self.gate(gate_input)  # (B, C, H, W)
        fused_feat = gate_weight * rgb_text_feat + (1 - gate_weight) * depth_text_feat
        
        # 6. 计算融合损失
        losses = {
            # 语义-像素损失：RGB与深度特征的一致性（间接关联文本语义）
            "pixel_loss": self.semantic_pixel_loss(rgb_c4, depth_c4),
            # 深度-语义损失：深度特征与RGB特征的差异约束（基于文本深度描述）
            "depth_loss": self.depth_semantic_loss(depth_c4, rgb_c4.detach())
        }
        return fused_feat, losses


class MultiScaleOptimizer(nn.Module):
    """
    多尺度特征优化模块（动态适配不同ConvNext型号的特征维度）
    融合c2（细节）、c3（结构）、c4（语义）特征，提升小目标/边界检测精度
    """
    def __init__(self, c2_dim, c3_dim, c4_dim, out_dim=256):
        super().__init__()
        self.out_dim = out_dim  # 输出特征维度（统一为256，平衡精度与速度）
        
        # 1. 各尺度特征通道数统一（c2→256, c3→256, c4→256）
        self.scale_c2 = self._build_scale_layer(c2_dim, out_dim)
        self.scale_c3 = self._build_scale_layer(c3_dim, out_dim)
        self.scale_c4 = self._build_scale_layer(c4_dim, out_dim)
        
        # 2. 跨尺度注意力（融合不同尺度信息）
        self.cross_attn = nn.MultiheadAttention(out_dim, num_heads=8, batch_first=True)
        
        # 3. 特征精炼（减少噪声，增强判别性）
        self.refine = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim)
        )
        # 初始化精炼层
        for m in self.refine.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _build_scale_layer(self, in_dim, out_dim):
        """构建单尺度特征通道调整层（Conv+BN+ReLU）"""
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, fused_c4, multi_scale_feats):
        """
        Args:
            fused_c4: (B, c4_dim, H4, W4) - 跨模态融合后的c4特征
            multi_scale_feats: dict - 包含rgb/depth的c2/c3特征
        Returns:
            optimized_feat: (B, out_dim, H2, W2) - 优化后的多尺度特征
        """
        # 1. 提取各尺度原始特征
        rgb_c2 = multi_scale_feats['rgb']['c2']
        rgb_c3 = multi_scale_feats['rgb']['c3']
        depth_c2 = multi_scale_feats['depth']['c2']
        depth_c3 = multi_scale_feats['depth']['c3']
        
        # 2. 通道统一与分辨率对齐（以c2分辨率为基准，最高分辨率）
        H2, W2 = rgb_c2.shape[2:]
        
        # c2特征处理（已为最高分辨率）
        rgb_c2 = self.scale_c2(rgb_c2)  # (B, 256, H2, W2)
        depth_c2 = self.scale_c2(depth_c2)
        
        # c3特征处理（上采样至c2分辨率）
        rgb_c3 = self.scale_c3(rgb_c3)
        rgb_c3_up = F.interpolate(rgb_c3, size=(H2, W2), mode='bilinear', align_corners=True)
        depth_c3 = self.scale_c3(depth_c3)
        depth_c3_up = F.interpolate(depth_c3, size=(H2, W2), mode='bilinear', align_corners=True)
        
        # c4融合特征处理（上采样至c2分辨率）
        fused_c4 = self.scale_c4(fused_c4)
        fused_c4_up = F.interpolate(fused_c4, size=(H2, W2), mode='bilinear', align_corners=True)
        
        # 3. 跨尺度注意力融合（以fused_c4为查询，c3/c2为键值）
        B, C, H, W = fused_c4_up.shape
        # 展平空间维度（适配多头注意力输入）
        q = fused_c4_up.flatten(2).permute(0, 2, 1)  # (B, H*W, 256) - 查询（语义主导）
        k = torch.cat([rgb_c3_up, depth_c3_up], dim=1).flatten(2).permute(0, 2, 1)  # (B, H*W, 512)→(B, H*W, 256)（降维）
        k = k[:, :, :C]  # 确保键维度与查询一致
        v = torch.cat([rgb_c2, depth_c2], dim=1).flatten(2).permute(0, 2, 1)  # (B, H*W, 512)→(B, H*W, 256)
        v = v[:, :, :C]
        
        attn_out, _ = self.cross_attn(q, k, v)  # (B, H*W, 256)
        attn_feat = attn_out.permute(0, 2, 1).view(B, C, H, W)  # 恢复空间维度
        
        # 4. 多特征融合与精炼
        combined_feat = attn_feat + rgb_c2 + depth_c2 + rgb_c3_up + depth_c3_up + fused_c4_up
        optimized_feat = self.refine(combined_feat)  # 减少冗余噪声
        
        return optimized_feat


class SalientPredictor(nn.Module):
    """
    显著性预测头（从256维特征→1通道显著性图）
    采用逐步上采样，提升边界精度
    """
    def __init__(self, in_dim=256, target_scale=4):
        super().__init__()
        self.target_scale = target_scale  # 总上采样倍数（适配c2到原始图像分辨率）
        
        # 预测网络（256→128→64→1，逐步降维+上采样）
        self.predict = nn.Sequential(
            # 第一阶段：降维+上采样（256→128，2x上采样）
            nn.Conv2d(in_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # 第二阶段：降维+上采样（128→64，2x上采样）
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # 输出层：64→1，sigmoid归一化
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        # 初始化预测头
        for m in self.predict.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feat, orig_size):
        """
        Args:
            feat: (B, 256, H, W) - 优化后的多尺度特征
            orig_size: tuple - 原始图像尺寸 (H_orig, W_orig)
        Returns:
            sal_map: (B, 1, H_orig, W_orig) - 显著性图
        """
        # 初步预测（上采样4x）
        pred = self.predict(feat)  # (B, 1, 4H, 4W)
        
        # 精确匹配原始图像分辨率（避免尺度偏差）
        sal_map = F.interpolate(pred, size=orig_size, mode='bilinear', align_corners=True)
        return sal_map


class TagNet(nn.Module):
    """
    TagNet: Text-Answer-Guided Network
    最终模型：FastVLM（离线生成文本）+ DinoV3-ConvNext（特征提取）+ CLIP（文本编码）
    仅训练：depth_adapter + 跨模态融合 + 多尺度优化 + 预测头
    """
    def __init__(self, convnext_model_name='convnext_tiny'):
        super().__init__()
        # 1. 初始化子模块（按依赖顺序）
        self.visual_encoder = DinoV3ConvNext(model_name=convnext_model_name)
        self.text_encoder = CLIPTextEncoder()
        
        # 动态获取维度配置（适配不同ConvNext型号）
        embed_dims = self.visual_encoder.cur_embed_dims  # [c1, c2, c3, c4]
        visual_c4_dim = embed_dims[-1]  # c4维度（视觉高层特征）
        text_dim = self.text_encoder.output_dim  # CLIP文本维度（768）
        
        # 2. 跨模态融合（适配动态维度）
        self.cross_modal_fusion = CrossModalFusion(visual_dim=visual_c4_dim, text_dim=text_dim)
        
        # 3. 多尺度优化（适配c2/c3/c4维度）
        self.multi_scale_opt = MultiScaleOptimizer(
            c2_dim=embed_dims[1],
            c3_dim=embed_dims[2],
            c4_dim=embed_dims[3],
            out_dim=256
        )
        
        # 4. 显著性预测头
        self.sal_predictor = SalientPredictor(in_dim=256)
        
        # 5. 损失函数（全局损失，权重经验值）
        self.bce_loss = nn.BCELoss()  # 像素级分类损失
        self.iou_loss = lambda x, y: 1 - torch.mean(  # IoU损失（提升区域一致性）
            (x * y).sum(dim=[1,2,3]) / ((x + y - x * y).sum(dim=[1,2,3]) + 1e-8)
        )
        self.semantic_consist_loss = nn.CosineEmbeddingLoss()  # 语义-视觉一致性损失
        
        # 6. 语义投影层（文本特征→256维，与优化后特征匹配）- 关键修正：移到__init__避免重复初始化
        self.text2feat_proj = nn.Linear(text_dim, 256)
        nn.init.kaiming_normal_(self.text2feat_proj.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.text2feat_proj.bias, 0)

    def forward(self, rgb, depth, texts, gt=None):
        """
        Args:
            rgb: (B, 3, H_orig, W_orig) - 原始RGB图像
            depth: (B, 1, H_orig, W_orig) - 原始深度图
            texts: list[str] - 离线FastVLM生成的显著性文本（长度=B）
            gt: (B, 1, H_orig, W_orig) - 显著性标签（训练时传入）
        Returns:
            outputs: dict - 包含预测图、损失（训练）
        """
        B, _, H_orig, W_orig = rgb.shape
        outputs = {}
        
        # 1. 特征提取（RGB+深度）
        visual_feats = self.visual_encoder(rgb, depth)
        
        # 2. 文本编码（CLIP，无梯度）
        text_feats = self.text_encoder(texts)  # (B, 768)
        
        # 关键修改：将文本特征从 Half 转为 Float，与线性层权重 dtype 一致 
        text_feats = text_feats.float()  # 添加这一行
        
        # 3. 跨模态融合（c4特征 + 文本特征）
        fused_c4, fusion_losses = self.cross_modal_fusion(visual_feats, text_feats)
        
        # 4. 多尺度特征优化（c2+c3+fused_c4）
        optimized_feat = self.multi_scale_opt(fused_c4, visual_feats)  # (B, 256, H2, W2)
        
        # 5. 显著性预测（匹配原始图像尺寸）
        sal_map = self.sal_predictor(optimized_feat, orig_size=(H_orig, W_orig))
        outputs['sal_map'] = sal_map
        
        # 6. 训练阶段：计算总损失
        if self.training and gt is not None:
            # 6.1 基础检测损失（BCE + IoU）
            bce_loss = self.bce_loss(sal_map, gt)
            iou_loss = self.iou_loss(sal_map, gt)
            base_loss = bce_loss + iou_loss
            
            # 6.2 语义一致性损失（优化后特征 ↔ 文本特征）
            # 特征压缩：(B,256,H,W)→(B,256)
            feat_compress = F.adaptive_avg_pool2d(optimized_feat, (1, 1)).squeeze(-1).squeeze(-1)
            # 文本投影：(B,768)→(B,256)
            text_proj = self.text2feat_proj(text_feats)
            # 余弦一致性损失（标签=1表示正相关）
            semantic_loss = self.semantic_consist_loss(feat_compress, text_proj, torch.ones(B, device=device))
            
            # 6.3 总损失（权重平衡各任务）
            total_loss = (
                base_loss + 
                0.3 * semantic_loss +  # 语义引导权重
                0.2 * fusion_losses['pixel_loss'] +  # 像素一致性权重
                0.2 * fusion_losses['depth_loss']    # 深度一致性权重
            )
            
            # 整理损失字典
            outputs['losses'] = {
                'base_loss': base_loss,
                'semantic_loss': semantic_loss,
                'pixel_loss': fusion_losses['pixel_loss'],
                'depth_loss': fusion_losses['depth_loss'],
                'total_loss': total_loss
            }
        
        return outputs


# ------------------------------ 训练/测试示例（修正逻辑错误）------------------------------
def train_step(model, dataloader, optimizer, epoch):
    """单轮训练（模拟真实训练流程）"""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (rgb, depth, texts, gt) in enumerate(dataloader):
        # 数据设备迁移
        rgb = rgb.to(device)
        depth = depth.to(device)
        gt = gt.to(device)
        
        # 前向传播
        outputs = model(rgb, depth, texts, gt)
        loss = outputs['losses']['total_loss']
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪防爆炸
        optimizer.step()
        
        # 损失统计
        total_loss += loss.item() * rgb.size(0)
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader.dataset)
    print(f"Epoch [{epoch+1}] Average Train Loss: {avg_loss:.4f}")
    return avg_loss


def test_step(model, dataloader):
    """测试推理（无梯度计算）"""
    model.eval()
    sal_maps = []
    texts_used = []
    
    with torch.no_grad():
        for rgb, depth, texts in dataloader:
            rgb = rgb.to(device)
            depth = depth.to(device)
            
            # 推理
            outputs = model(rgb, depth, texts)
            sal_maps.append(outputs['sal_map'].cpu())
            texts_used.extend(texts)
    
    # 整理输出
    sal_maps = torch.cat(sal_maps, dim=0)
    print(f"Test Completed. Salient Maps Shape: {sal_maps.shape}, Texts Count: {len(texts_used)}")
    return sal_maps, texts_used


# ------------------------------ 模拟数据加载器（示例）------------------------------
class MockRGBDDataset(torch.utils.data.Dataset):
    """模拟RGB-D显著性数据集（含离线FastVLM文本）"""
    def __init__(self, size=100, img_size=(224, 224)):
        self.size = size
        self.img_size = img_size
        # 模拟离线FastVLM生成的显著性文本（真实场景需替换为实际生成结果）
        self.text_pool = [
            "A red apple in the center of the image",
            "A blue cup on the table with shallow depth",
            "A white book on the shelf, deeper than the background",
            "A black pen on the desk, salient and small"
        ]
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 模拟RGB图像（随机值，真实场景需加载数据）
        rgb = torch.randn(3, *self.img_size)
        # 模拟深度图（随机值，真实场景需加载数据）
        depth = torch.randn(1, *self.img_size)
        # 随机选择文本（真实场景需与图像一一对应）
        text = self.text_pool[idx % len(self.text_pool)]
        # 模拟GT（随机值，真实场景需加载标签）
        gt = torch.rand(1, *self.img_size)
        
        return rgb, depth, text, gt


if __name__ == "__main__":
    # 1. 配置参数
    img_size = (224, 224)
    batch_size = 2
    epochs = 3
    lr = 1e-4
    
    # 2. 构建数据集与加载器
    train_dataset = MockRGBDDataset(size=100, img_size=img_size)
    test_dataset = MockRGBDDataset(size=20, img_size=img_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. 初始化模型与优化器（仅优化可训练参数）
    model = TagNet(convnext_model_name='convnext_small').to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # 学习率调度
    
    # 4. 训练与测试
    print("="*50, "Start Training", "="*50)
    for epoch in range(epochs):
        train_avg_loss = train_step(model, train_loader, optimizer, epoch)
        scheduler.step()  # 更新学习率
    
    print("\n" + "="*50, "Start Testing", "="*50)
    test_sal_maps, test_texts = test_step(model, test_loader)
    
    # 5. 保存模型（仅保存可训练参数，减少存储）
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epochs
    }, 'TagNet.pth')
    print("\nModel Saved to 'fastvlm_dinov3_clip_sod.pth'")
