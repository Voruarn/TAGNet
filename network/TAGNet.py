import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoModel, AutoProcessor
import clip
from PIL import Image
import numpy as np
from network.convnext import convnext_tiny, convnext_small, convnext_base, Block
from network.Modules import VisualTextAttention, Bottleneck, DBlock, ModalEnhancedFusion


# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvNextModel(nn.Module):
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
        
        self.depth_adapter = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)

        nn.init.kaiming_normal_(self.depth_adapter.weight, mode='fan_out', nonlinearity='relu')
        if self.depth_adapter.bias is not None:
            nn.init.constant_(self.depth_adapter.bias, 0)

    def forward(self, rgb, depth):
        depth_3ch = self.depth_adapter(depth)  # (B, 3, H, W)
        
        rgb_c1, rgb_c2, rgb_c3, rgb_c4 = self.convnext(rgb)
        depth_c1, depth_c2, depth_c3, depth_c4 = self.convnext(depth_3ch)
        
        return {
            'rgb':  [rgb_c1, rgb_c2, rgb_c3, rgb_c4],
            'depth': [depth_c1, depth_c2, depth_c3, depth_c4]
        }


class CLIPTextEncoder(nn.Module):
    def __init__(self, model_name="ViT-B/16"):
        super().__init__()

        self.clip_model, self.clip_preprocess = clip.load(model_name, device=device)

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
        text_tokens = clip.tokenize(texts, truncate=True).to(device)
    
        with torch.no_grad():
            text_feats = self.clip_model.encode_text(text_tokens)
 
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        return text_feats
        

class SalientPredictor(nn.Module):
    def __init__(self, in_dim=256):
        super().__init__()

        self.predict = nn.Sequential(
            nn.Conv2d(in_dim, in_dim//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_dim//2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(in_dim//2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        for m in self.predict.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feat, orig_size):
        pred = self.predict(feat)  # (B, 1, 4H, 4W)
        sal_map = F.interpolate(pred, size=orig_size, mode='bilinear', align_corners=True)
        return sal_map


class CrossModalFusion(nn.Module):
    def __init__(self, vis_dims, text_dim):
        super().__init__()

        self.vt_attn1 = VisualTextAttention(vis_dims[0], text_dim)
        self.vt_attn2 = VisualTextAttention(vis_dims[1], text_dim)
        self.vt_attn3 = VisualTextAttention(vis_dims[2], text_dim)
        self.vt_attn4 = VisualTextAttention(vis_dims[3], text_dim)

        self.rgbd_fuse1 = ModalEnhancedFusion(vis_dims[0])
        self.rgbd_fuse2 = ModalEnhancedFusion(vis_dims[1])
        self.rgbd_fuse3 = ModalEnhancedFusion(vis_dims[2])
        self.rgbd_fuse4 = ModalEnhancedFusion(vis_dims[3])

        self.dec1 = Bottleneck(vis_dims[1] + vis_dims[0], vis_dims[0])
        self.dec2 = Bottleneck(vis_dims[2] + vis_dims[1], vis_dims[1])
        self.dec3 = Bottleneck(vis_dims[3] + vis_dims[2], vis_dims[2])
        self.dec4 = Bottleneck(vis_dims[3], vis_dims[3])
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, visual_feats, text):
        c1,c2,c3,c4 = visual_feats['rgb']
        d1,d2,d3,d4 = visual_feats['depth']

        cd1 = self.rgbd_fuse1(c1, d1)
        cd2 = self.rgbd_fuse2(c2, d2)
        cd3 = self.rgbd_fuse3(c3, d3)
        cd4 = self.rgbd_fuse4(c4, d4)

        feat1 = self.vt_attn1(cd1, text)
        feat2 = self.vt_attn2(cd2, text)
        feat3 = self.vt_attn3(cd3, text)
        feat4 = self.vt_attn4(cd4, text)

        up4 = self.dec4(feat4)

        up3 = torch.cat([feat3, self.upsample2(up4)], dim=1)
        up3 = self.dec3(up3)

        up2 = torch.cat([feat2, self.upsample2(up3)], dim=1)
        up2 = self.dec2(up2)

        up1 = torch.cat([feat1, self.upsample2(up2)], dim=1)
        up1 = self.dec1(up1)
        return up1


class TAGNet(nn.Module):
    ### TAGNet: Text-Answer-Guided Network 
    def __init__(self, convnext_model_name='convnext_base', text_dim=512):
        super().__init__()

        self.visual_encoder = ConvNextModel(model_name=convnext_model_name)
       
        vis_dims = self.visual_encoder.cur_embed_dims 

        self.cross_modal_fusion = CrossModalFusion(vis_dims=vis_dims, text_dim=text_dim)
        self.predictor = SalientPredictor(vis_dims[0])

        self.bce_loss = nn.BCELoss()  
        self.iou_loss = lambda x, y: 1 - torch.mean(  
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
