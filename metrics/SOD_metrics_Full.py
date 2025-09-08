import os
import time
import numpy as np
import torch
from torchvision import transforms
import scipy
import scipy.ndimage
from tqdm import tqdm

# 全局阈值配置（与参考代码保持一致）
threshold_sal, upper_sal, lower_sal = 0.5, 1, 0


class _StreamMetrics(object):
    def __init__(self):
        raise NotImplementedError()

    def update(self, gt, pred):
        raise NotImplementedError()

    def get_results(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()      


class SODMetrics(_StreamMetrics):
    """
    优化后的RGB-D SOD任务指标计算器
    支持指标：MAE、Fmeasure（max/mean）、Smeasure、Emeasure（max/mean/adaptive）
    计算逻辑完全对齐参考的Eval_thread代码
    """
    def __init__(self, cuda=False):
        self.cuda = cuda
        self.beta2 = 0.3  # Fmeasure的β²值（参考行业标准）
        self.num_thresh = 255  # 阈值数量（Fmeasure/Emeasure用255个阈值）
        
        # -------------------------- 初始化指标累计变量 --------------------------
        # 1. MAE相关
        self.mae_sum = 0.0  # 所有样本的MAE累加和
        self.mae_valid_cnt = 0  # MAE有效样本数（排除NaN）
        
        # 2. Smeasure相关
        self.sm_sum = 0.0  # 所有样本的S值累加和
        self.sm_valid_cnt = 0  # Smeasure有效样本数（排除NaN）
        
        # 3. Fmeasure相关（仅统计GT非全黑的样本）
        self.fm_prec_total = torch.zeros(self.num_thresh, device='cuda' if cuda else 'cpu')  # 所有有效样本的Prec累加
        self.fm_recall_total = torch.zeros(self.num_thresh, device='cuda' if cuda else 'cpu')  # 所有有效样本的Recall累加
        self.fm_valid_cnt = 0  # Fmeasure有效样本数（GT非全黑）
        
        # 4. Emeasure相关（仅统计GT非全黑/非全白的样本）
        self.em_score_total = torch.zeros(self.num_thresh, device='cuda' if cuda else 'cpu')  # 所有有效样本的Em累加
        self.em_adp_sum = 0.0  # 所有有效样本的自适应Em累加
        self.em_valid_cnt = 0  # Emeasure有效样本数（GT非全黑/非全白）

    def update(self, preds, labels):
        """
        更新指标累计：单次输入一个batch的预测图和标签
        Args:
            preds: (B, 1, H, W) - 模型输出的显著性图（值范围[0,1]）
            labels: (B, 1, H, W) - 真值标签（值范围[0,1]，需提前归一化）
        """
        # 确保输入为4D张量（B,1,H,W），并适配设备
        if preds.dim() != 4 or labels.dim() != 4:
            raise ValueError(f"输入必须为4D张量(B,1,H,W)，当前preds维度：{preds.dim()}, labels维度：{labels.dim()}")
        if self.cuda and preds.device != torch.device('cuda'):
            preds = preds.cuda()
            labels = labels.cuda()
        
        # 逐样本计算指标（batch内循环）
        for pred, gt in zip(preds, labels):
            # 挤压通道维度（(1,H,W)→(H,W)），方便计算
            pred = pred.squeeze(0)
            gt = gt.squeeze(0)
            
            # 1. 计算MAE
            self._cal_mae_single(pred, gt)
            # 2. 计算Smeasure
            self._cal_sm_single(pred, gt, alpha=0.5)
            # 3. 计算Fmeasure（仅GT非全黑时计算）
            if not torch.allclose(gt.mean(), torch.tensor(0.0, device=gt.device)):
                self._cal_fm_single(pred, gt)
            # 4. 计算Emeasure（仅GT非全黑/非全白时计算）
            gt_mean = gt.mean()
            if not (torch.allclose(gt_mean, torch.tensor(0.0, device=gt.device)) or 
                    torch.allclose(gt_mean, torch.tensor(1.0, device=gt.device))):
                self._cal_em_single(pred, gt)

    def _cal_mae_single(self, pred, gt):
        """单样本MAE计算（参考Eval_thread.Eval_MAE）"""
        mae = torch.abs(pred - gt).mean()
        if not torch.isnan(mae):  # 排除NaN值
            self.mae_sum += mae.item()
            self.mae_valid_cnt += 1

    def _cal_sm_single(self, pred, gt, alpha=0.5):
        """单样本Smeasure计算（参考Eval_thread.Eval_Smeasure）"""
        # 先将GT二值化（与参考代码一致）
        gt_bin = torch.where(gt >= 0.5, torch.tensor(1.0, device=gt.device), torch.tensor(0.0, device=gt.device))
        gt_mean = gt_bin.mean()
        
        # 处理特殊情况
        if torch.allclose(gt_mean, torch.tensor(0.0, device=gt.device)):
            # GT全黑：S = 1 - 预测图均值
            Q = 1.0 - pred.mean().item()
        elif torch.allclose(gt_mean, torch.tensor(1.0, device=gt.device)):
            # GT全白：S = 预测图均值
            Q = pred.mean().item()
        else:
            # 正常情况：S = α*S_object + (1-α)*S_region
            S_obj = self._S_object(pred, gt_bin)
            S_reg = self._S_region(pred, gt_bin)
            Q = (alpha * S_obj + (1 - alpha) * S_reg).item()
        
        # 排除NaN值，更新累计
        if not np.isnan(Q):
            self.sm_sum += Q
            self.sm_valid_cnt += 1

    def _cal_fm_single(self, pred, gt):
        """单样本Fmeasure计算（参考Eval_thread.Eval_Fmeasure）"""
        # 计算当前样本在255个阈值下的Prec和Recall
        prec, recall = self._eval_pr(pred, gt, self.num_thresh)
        # 累加至全局Prec/Recall
        self.fm_prec_total += prec
        self.fm_recall_total += recall
        self.fm_valid_cnt += 1

    def _cal_em_single(self, pred, gt):
        """单样本Emeasure计算（参考Eval_thread.Eval_Emeasure）"""
        # 计算255个阈值下的Em分数
        em_score = self._eval_e(pred, gt, self.num_thresh)
        # 计算自适应阈值的Em分数（阈值=2*预测图均值）
        em_adp = self._eval_adp_e(pred, gt)
        
        # 累加至全局
        self.em_score_total += em_score
        self.em_adp_sum += em_adp.item()
        self.em_valid_cnt += 1

    def get_results(self):
        """
        返回所有指标的最终结果（确保无设备依赖，返回Python数值）
        Returns:
            dict: 包含MAE、Fmeasure(max/mean)、Smeasure、Emeasure(max/mean/adaptive)
        """
        # -------------------------- 1. 计算MAE --------------------------
        mae = self.mae_sum / self.mae_valid_cnt if self.mae_valid_cnt > 0 else 0.0
        
        # -------------------------- 2. 计算Smeasure --------------------------
        sm = self.sm_sum / self.sm_valid_cnt if self.sm_valid_cnt > 0 else 0.0
        
        # -------------------------- 3. 计算Fmeasure --------------------------
        max_f, mean_f = 0.0, 0.0
        if self.fm_valid_cnt > 0:
            # 全局Prec/Recall平均（所有有效样本的平均值）
            avg_prec = self.fm_prec_total / self.fm_valid_cnt
            avg_recall = self.fm_recall_total / self.fm_valid_cnt
            # 计算Fscore：(1+β²)*P*R/(β²*P + R)
            f_score = (1 + self.beta2) * avg_prec * avg_recall / (self.beta2 * avg_prec + avg_recall + 1e-20)
            f_score[torch.isnan(f_score)] = 0.0  # 排除NaN
            max_f = f_score.max().item()
            mean_f = f_score.mean().item()
        
        # -------------------------- 4. 计算Emeasure --------------------------
        max_e, mean_e, adp_e = 0.0, 0.0, 0.0
        if self.em_valid_cnt > 0:
            # 全局Em分数平均（所有有效样本的平均值）
            avg_em_score = self.em_score_total / self.em_valid_cnt
            avg_em_score[torch.isnan(avg_em_score)] = 0.0  # 排除NaN
            max_e = avg_em_score.max().item()
            mean_e = avg_em_score.mean().item()
            # 自适应Em（所有有效样本的平均值）
            # adp_e = self.em_adp_sum / self.em_valid_cnt
        
        # 返回最终结果（统一为Python float，避免张量依赖）
        return {
            "MAE": round(mae, 4),
            "Fmeasure": {
                "max": round(max_f, 4),
                "mean": round(mean_f, 4)
            },
            "Smeasure": round(sm, 4),
            "Emeasure": {
                "max": round(max_e, 4),
                "mean": round(mean_e, 4),
            }
        }

    def reset(self):
        """重置所有累计变量，用于新一轮评估"""
        # MAE相关
        self.mae_sum = 0.0
        self.mae_valid_cnt = 0
        
        # Smeasure相关
        self.sm_sum = 0.0
        self.sm_valid_cnt = 0
        
        # Fmeasure相关
        self.fm_prec_total = torch.zeros(self.num_thresh, device='cuda' if self.cuda else 'cpu')
        self.fm_recall_total = torch.zeros(self.num_thresh, device='cuda' if self.cuda else 'cpu')
        self.fm_valid_cnt = 0
        
        # Emeasure相关
        self.em_score_total = torch.zeros(self.num_thresh, device='cuda' if self.cuda else 'cpu')
        self.em_adp_sum = 0.0
        self.em_valid_cnt = 0

    # -------------------------- 内部辅助函数（对齐Eval_thread） --------------------------
    def _S_object(self, pred, gt):
        """计算Smeasure的Object项（参考Eval_thread._S_object）"""
        fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
        bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
        o_fg = self._object(fg, gt)
        o_bg = self._object(bg, 1 - gt)
        u = gt.mean().item()
        return u * o_fg + (1 - u) * o_bg

    def _object(self, pred, gt):
        """计算Object项的子函数（参考Eval_thread._object）"""
        temp = pred[gt == 1]
        if temp.numel() == 0:
            return 0.0
        x = temp.mean().item()
        sigma_x = temp.std().item()
        return 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)

    def _S_region(self, pred, gt):
        """计算Smeasure的Region项（参考Eval_thread._S_region）"""
        X, Y = self._centroid(gt)  # 获取GT的质心
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self._divideGT(gt, X, Y)  # 划分GT为4个区域
        p1, p2, p3, p4 = self._dividePrediction(pred, X, Y)  # 划分预测图为4个区域
        # 计算每个区域的SSIM
        Q1 = self._ssim(p1, gt1)
        Q2 = self._ssim(p2, gt2)
        Q3 = self._ssim(p3, gt3)
        Q4 = self._ssim(p4, gt4)
        # 加权求和
        return w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4

    def _centroid(self, gt):
        """计算GT的质心（参考Eval_thread._centroid）"""
        rows, cols = gt.size()
        if torch.allclose(gt.sum(), torch.tensor(0.0, device=gt.device)):
            # GT全黑：质心设为图像中心
            X = torch.tensor(round(cols / 2), dtype=torch.long, device=gt.device)
            Y = torch.tensor(round(rows / 2), dtype=torch.long, device=gt.device)
        else:
            total = gt.sum().item()
            i = torch.arange(cols, dtype=torch.float, device=gt.device)
            j = torch.arange(rows, dtype=torch.float, device=gt.device)
            # 计算x方向质心（列）
            X = torch.round((gt.sum(dim=0) * i).sum() / total).long()
            # 计算y方向质心（行）
            Y = torch.round((gt.sum(dim=1) * j).sum() / total).long()
        return X, Y

    def _divideGT(self, gt, X, Y):
        """划分GT为4个区域（参考Eval_thread._divideGT）"""
        h, w = gt.size()
        area = h * w
        # 划分4个区域：左上(LT)、右上(RT)、左下(LB)、右下(RB)
        LT = gt[:Y, :X]
        RT = gt[:Y, X:]
        LB = gt[Y:, :X]
        RB = gt[Y:, X:]
        # 计算每个区域的权重
        w1 = (X * Y) / area
        w2 = ((w - X) * Y) / area
        w3 = (X * (h - Y)) / area
        w4 = 1.0 - w1 - w2 - w3
        return LT, RT, LB, RB, w1, w2, w3, w4

    def _dividePrediction(self, pred, X, Y):
        """划分预测图为4个区域（参考Eval_thread._dividePrediction）"""
        h, w = pred.size()
        LT = pred[:Y, :X]
        RT = pred[:Y, X:]
        LB = pred[Y:, :X]
        RB = pred[Y:, X:]
        return LT, RT, LB, RB

    def _ssim(self, pred, gt):
        """计算两个区域的SSIM（参考Eval_thread._ssim）"""
        gt = gt.float()
        h, w = pred.size()
        N = h * w
        if N == 0:
            return 0.0
        # 计算均值
        x = pred.mean().item()
        y = gt.mean().item()
        # 计算方差和协方差
        sigma_x2 = ((pred - x) ** 2).sum().item() / (N - 1 + 1e-20)
        sigma_y2 = ((gt - y) ** 2).sum().item() / (N - 1 + 1e-20)
        sigma_xy = ((pred - x) * (gt - y)).sum().item() / (N - 1 + 1e-20)
        # 计算SSIM
        alpha = 4 * x * y * sigma_xy
        beta = (x ** 2 + y ** 2) * (sigma_x2 + sigma_y2)
        if alpha != 0:
            return alpha / (beta + 1e-20)
        elif alpha == 0 and beta == 0:
            return 1.0
        else:
            return 0.0

    def _eval_pr(self, y_pred, y, num):
        """计算Precision和Recall（参考Eval_thread._eval_pr）"""
        thlist = torch.linspace(0, 1 - 1e-10, num, device=y_pred.device)
        prec = torch.zeros(num, device=y_pred.device)
        recall = torch.zeros(num, device=y_pred.device)
        for i in range(num):
            # 按当前阈值二值化预测图
            y_temp = (y_pred >= thlist[i]).float()
            # 计算TP（真阳性）
            tp = (y_temp * y).sum()
            # 计算Precision和Recall（加1e-20避免除零）
            prec[i] = tp / (y_temp.sum() + 1e-20)
            recall[i] = tp / (y.sum() + 1e-20)
        return prec, recall

    def _eval_e(self, y_pred, y, num):
        """计算Emeasure的255个阈值分数（参考Eval_thread._eval_e）"""
        thlist = torch.linspace(0, 1 - 1e-10, num, device=y_pred.device)
        em_score = torch.zeros(num, device=y_pred.device)
        for i in range(num):
            # 按当前阈值二值化预测图
            y_temp = (y_pred >= thlist[i]).float()
            # 计算对齐矩阵（Enhanced Alignment Matrix）
            fm = y_temp - y_temp.mean()
            gt = y - y.mean()
            align_matrix = 2 * gt * fm / (gt ** 2 + fm ** 2 + 1e-20)
            # 计算Enhanced分数
            enhanced = ((align_matrix + 1) ** 2) / 4
            em_score[i] = enhanced.sum() / (y.numel() - 1 + 1e-20)
        return em_score

    def _eval_adp_e(self, y_pred, y):
        """计算自适应阈值的Emeasure（参考Eval_thread._eval_adp_e）"""
        # 自适应阈值：2*预测图均值（不超过1）
        thr = y_pred.mean() * 2
        thr = torch.clamp(thr, max=1.0)
        # 二值化预测图
        y_temp = (y_pred >= thr).float()
        # 计算对齐矩阵
        fm = y_temp - y_temp.mean()
        gt = y - y.mean()
        align_matrix = 2 * gt * fm / (gt ** 2 + fm ** 2 + 1e-20)
        # 计算Enhanced分数
        enhanced = ((align_matrix + 1) ** 2) / 4
        return enhanced.sum() / (y.numel() - 1 + 1e-20)
