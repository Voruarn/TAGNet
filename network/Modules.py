import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from network.convnext import Block

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.ReLU()  # default activation
    # default_act=nn.ReLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
    

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    

class DBlock(nn.Module):
    # DBlock
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), 1)
        return self.cv2(torch.cat((self.m(a), b), 1))
    
class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4, norm_cfg=dict(type="BN", requires_grad=True)):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.pos(x) + x
        x = x.permute(0, 2, 3, 1)
        x = self.act(x)
        x = self.fc2(x)

        return x


class VisualTextAttention(nn.Module):
    """视觉-文本交叉注意力模块（对应图c）- 修复维度不匹配问题"""
    def __init__(self, visual_dim, text_dim,):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        
        # 新增：文本特征维度投影层（text_dim→visual_dim）
        self.text_proj = nn.Linear(text_dim, visual_dim)
        # Cross-Attention组件（embed_dim=visual_dim，与投影后文本维度一致）
        self.cross_attn = nn.MultiheadAttention(visual_dim, num_heads=8, batch_first=True)
        
        # 后续变换层：Linear→ReLU→Linear→Tanh
        self.linear1 = nn.Linear(visual_dim, visual_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(visual_dim, visual_dim)
        self.tanh = nn.Tanh()

        # 初始化（包含新增的text_proj）
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, visual_feat, text_feat):
        """
        visual_feat: (B, H, W, visual_dim) - 带空间维度的视觉特征（4D）
        text_feat: (B, text_dim) - 文本特征
        """
        # 1. 保存视觉特征的空间维度信息（用于后续恢复形状）
        B, C, H, W = visual_feat.shape 
        visual_feat=visual_feat.permute(0, 2, 3, 1)  # 获取 B, H, W, C
        visual_feat_flat = visual_feat.reshape(B, H*W, C)  # 展平空间维度
        
        # 3. 文本特征投影与扩展（保持不变）
        text_feat_proj = self.text_proj(text_feat)  # (B, visual_dim)
        text_feat_expand = text_feat_proj.unsqueeze(1)  # (B, 1, visual_dim)
        
        # 4. Cross-Attention：输入为3D张量（已展平）
        attn_out, _ = self.cross_attn(visual_feat_flat, text_feat_expand, text_feat_expand)  # (B, H*W, C)
        
        # 5. 后续变换（保持不变）
        linear_out = self.linear1(attn_out)
        linear_out = self.relu(linear_out)
        linear_out = self.linear2(linear_out)
        linear_out = self.tanh(linear_out)
        
        # 6. 残差连接（使用展平的视觉特征）
        out = attn_out * linear_out
        out = out + visual_feat_flat  # (B, H*W, C)

        out = out.reshape(B, H, W, C)  # 恢复空间维度, 将3D序列恢复为4D空间特征 [B, H, W, C]
        out = out.permute(0, 3, 1, 2) # B, C, H, W

        return out

