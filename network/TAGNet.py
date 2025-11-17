import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoModel, AutoProcessor
import clip
from PIL import Image
import numpy as np
from network.convnext import convnext_tiny, convnext_small, convnext_base, Block
from network.Modules import VisualTextAttention, Bottleneck, DBlock


# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvNextModel(nn.Module):

    # 不同ConvNext型号的各阶段输出维度（与convnext.py中模型输出一致）
    embed_dims = {
        "convnext_tiny": [96, 192, 384, 768],    # c1, c2, c3, c4
        "convnext_small": [96, 192, 384, 768],
        "convnext_base": [128, 256, 512, 1024]
    }

    def __init__(self, model_name='convnext_base', pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.cur_embed_dims = self.embed_dims[model_name]  # 当前模型的维度配置
        
        self.convnext = eval(model_name)(pretrained=pretrained)
        
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
            'rgb':  [rgb_c1, rgb_c2, rgb_c3, rgb_c4],
            'depth': [depth_c1, depth_c2, depth_c3, depth_c4]
        }


class CLIPTextEncoder(nn.Module):
    """
    CLIP文本编码器（ViT-B/16）- 冻结参数
    输出： 512 维归一化文本特征
    """
    def __init__(self, model_name="ViT-B/16"):
        super().__init__()
        # 加载CLIP ViT-B/16（输出文本特征维度768）
        self.clip_model, self.clip_preprocess = clip.load(model_name, device=device)
        
        # 冻结所有参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self.output_dim = 512  # CLIP ViT-B/16文本特征维度

    def forward(self, texts):
        """
        Args:
            texts: list[str] - 显著性文本列表（长度=batch_size）
        Returns:
            text_feats: (B, 512) - 归一化文本特征
        """
        # CLIP Tokenize（自动处理padding和截断）
        text_tokens = clip.tokenize(texts, truncate=True).to(device)
        
        # 编码（无梯度计算）
        with torch.no_grad():
            text_feats = self.clip_model.encode_text(text_tokens)
        
        # 特征归一化（提升跨模态匹配稳定性）
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        return text_feats
        

class SalientPredictor(nn.Module):
    """
    显著性预测头（从256维特征→1通道显著性图）
    采用逐步上采样，提升边界精度
    """
    def __init__(self, in_dim=256):
        super().__init__()

        self.predict = nn.Sequential(
            nn.Conv2d(in_dim, in_dim//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_dim//2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # 输出层：64→1，sigmoid归一化
            nn.Conv2d(in_dim//2, 1, kernel_size=1),
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


class CrossModalFusion(nn.Module):
    """
    跨模态融合模块（RGB-文本 + 深度-文本双路径）
    适配动态视觉/文本维度，仅训练融合相关参数
    """
    def __init__(self, vis_dims, text_dim):
        super().__init__()

        self.vt_attn1 = VisualTextAttention(vis_dims[0], text_dim)
        self.vt_attn2 = VisualTextAttention(vis_dims[1], text_dim)
        self.vt_attn3 = VisualTextAttention(vis_dims[2], text_dim)
        self.vt_attn4 = VisualTextAttention(vis_dims[3], text_dim)

        self.dt_attn1 = VisualTextAttention(vis_dims[0], text_dim)
        self.dt_attn2 = VisualTextAttention(vis_dims[1], text_dim)
        self.dt_attn3 = VisualTextAttention(vis_dims[2], text_dim)
        self.dt_attn4 = VisualTextAttention(vis_dims[3], text_dim)

        self.dec1 = nn.Sequential(
            Bottleneck(vis_dims[1] + vis_dims[0]*2, vis_dims[0]),
            DBlock(vis_dims[0], vis_dims[0])
        )
        self.dec2 = nn.Sequential(
            Bottleneck(vis_dims[2] + vis_dims[1]*2, vis_dims[1]),
            DBlock(vis_dims[1], vis_dims[1])
        )
        self.dec3 = nn.Sequential(
            Bottleneck(vis_dims[3] + vis_dims[2]*2, vis_dims[2]),
            DBlock(vis_dims[2], vis_dims[2])
        )
        self.dec4 = nn.Sequential(
            Bottleneck(vis_dims[3]*2, vis_dims[3]),
            DBlock(vis_dims[3], vis_dims[3])
        )

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


    def forward(self, visual_feats, text):
        c1,c2,c3,c4 = visual_feats['rgb']
        d1,d2,d3,d4 = visual_feats['depth']

        feat1 = self.vt_attn1(c1, text)
        feat2 = self.vt_attn2(c2, text)
        feat3 = self.vt_attn3(c3, text)
        feat4 = self.vt_attn4(c4, text)

        df1 = self.dt_attn1(d1, text)
        df2 = self.dt_attn2(d2, text)
        df3 = self.dt_attn3(d3, text)
        df4 = self.dt_attn4(d4, text)

        up4 = torch.cat([feat4, df4], dim=1)
        up4 = self.dec4(up4)

        up3 = torch.cat([feat3, df3, self.upsample2(up4)], dim=1)
        up3 = self.dec3(up3)

        up2 = torch.cat([feat2, df2, self.upsample2(up3)], dim=1)
        up2 = self.dec2(up2)

        up1 = torch.cat([feat1, df1, self.upsample2(up2)], dim=1)
        up1 = self.dec1(up1)

        return up1


class TAGNet(nn.Module):
    """
    TAGNet: Text-Answer-Guided Network
    最终模型:   FastVLM（离线生成文本）+ ConvNext（视觉特征提取）+ CLIP（文本编码）
    仅训练:     ConvNext + 跨模态融合 + 多尺度优化 + 预测头
    """
    def __init__(self, convnext_model_name='convnext_base', text_dim=512):
        super().__init__()

        self.visual_encoder = ConvNextModel(model_name=convnext_model_name)
       
        vis_dims = self.visual_encoder.cur_embed_dims 

        self.cross_modal_fusion = CrossModalFusion(vis_dims=vis_dims, text_dim=text_dim)
        self.predictor = SalientPredictor(vis_dims[0])

        self.bce_loss = nn.BCELoss()  # 像素级分类损失
        self.iou_loss = lambda x, y: 1 - torch.mean(  # IoU损失（提升区域一致性）
            (x * y).sum(dim=[1,2,3]) / ((x + y - x * y).sum(dim=[1,2,3]) + 1e-8)
        )
      

    def forward(self, rgb, depth, texts, gt=None):
        B, _, H_orig, W_orig = rgb.shape
        outputs = {}
        
        visual_feats = self.visual_encoder(rgb, depth)
        
        fused = self.cross_modal_fusion(visual_feats, texts)
        sal_map = self.predictor(fused, orig_size=(H_orig, W_orig))

        outputs['sal_map'] = sal_map
        
        if self.training and gt is not None:
            bce_loss = self.bce_loss(sal_map, gt)
            iou_loss = self.iou_loss(sal_map, gt)
            total_loss = bce_loss + iou_loss
            
            outputs['losses'] = {
                'total_loss': total_loss
            }
        
        return outputs
