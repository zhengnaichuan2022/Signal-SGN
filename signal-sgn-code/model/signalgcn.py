"""
Signal-SGN: Spiking Graph Convolutional Network for Skeleton-based Action Recognition

Core modules:
- 1D-SGC: 1D Spiking Graph Convolution (unit_gcn, MSA_Conv)
- FSC: Frequency Spiking Convolution (ComplexConv2d, SpikingAttention)
- MWTF: Multi-Scale Wavelet Transform Feature Fusion (MultiWaveletTransform)
"""

import math
from spikingjelly.activation_based import neuron, layer, functional
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from spikingjelly.activation_based.encoding import *
from module.dwt import *


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


# ============ FSC Module: Frequency Spiking Convolution ============
class ComplexConv2d(nn.Module):
    """Complex convolution for frequency domain - real/imaginary branch interaction"""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_r = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm1d(out_channels))
        self.LIF_R = neuron.LIFNode(step_mode='m', backend='cupy')
        self.conv_i = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm1d(out_channels))
        self.LIF_I = neuron.LIFNode(step_mode='m', backend='cupy')
        self.conv_r_1 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size, stride),
            nn.BatchNorm1d(out_channels))
        self.conv_i_1 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size, stride),
            nn.BatchNorm1d(out_channels))
        self.LIF_i = neuron.LIFNode(step_mode='m', backend='cupy')
        self.LIF_r = neuron.LIFNode(step_mode='m', backend='cupy')

    def forward(self, input):
        T, N, C, V = input.shape
        input = input.flatten(0, 1)
        x1 = self.conv_r(input.real) - self.conv_i(input.imag)
        x1 = x1.reshape(T, N, -1, V).contiguous()
        x1 = self.LIF_R(x1).flatten(0, 1)
        x2 = self.conv_r(input.real) - self.conv_i(input.imag)
        x2 = x2.reshape(T, N, -1, V).contiguous()
        x2 = self.LIF_R(x2).flatten(0, 1)
        x_r = self.conv_r_1(x1).reshape(T, N, -1, V).contiguous()
        x_r = self.LIF_r(x_r)
        x_i = self.conv_i_1(x2).reshape(T, N, -1, V).contiguous()
        x_i = self.LIF_r(x_i)
        return x_r + x_i


class SpikingAttention(nn.Module):
    """FSC: Frequency Spiking Convolution with learnable window and complex modeling"""

    def __init__(self, *, in_channels=64, out_channel=64, num_point=25, Times=4,
                 dim=1, dim_head=64, heads=8, dropout=0.):
        super().__init__()
        self.proj_conv = nn.Conv1d(in_channels, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(out_channel)
        self.proj_lif = neuron.LIFNode(step_mode='m', backend='cupy')
        self.complexConv2d = ComplexConv2d(out_channel, out_channel)
        self.window_w = nn.Parameter(torch.randn(1, 1, num_point))

        conv_init(self.proj_conv)
        bn_init(self.proj_bn, 1)

    def forward(self, x):
        T, N, C, V = x.shape
        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, N, -1, V).contiguous()
        x = self.proj_lif(x)
        x = x * self.window_w
        x = torch.fft.fftn(x, dim=(3))
        x = self.complexConv2d(x)
        return x


# ============ 1D-SGC Module: Spiking Graph Convolution ============
class MSA_Conv(nn.Module):
    """Multi-head Spiking Attention for spatial modeling"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = neuron.LIFNode(step_mode='m', backend='cupy')
        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = neuron.LIFNode(step_mode='m', backend='cupy')
        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = neuron.LIFNode(step_mode='m', backend='cupy')
        self.attn_lif = neuron.LIFNode(step_mode='m', v_threshold=0.5, backend='cupy')
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = neuron.LIFNode(step_mode='m', backend='cupy')

    def forward(self, x):
        T, N, C, V = x.shape
        identity = x
        x_for_qkv = x.flatten(0, 1)
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, N, C, V).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, N, V, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, N, C, V).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, N, V, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, N, C, V).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, N, V, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        attn = (q @ k.transpose(-2, -1))
        x = (attn @ v) * self.scale
        x = x.transpose(3, 4).reshape(T, N, C, V).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x)).reshape(T, N, C, V))
        x = x + identity
        return x


class unit_gcn(nn.Module):
    """1D-SGC: 1D Spiking Graph Convolution unit"""

    def __init__(self, in_channels, out_channels, A, adaptive=True, Times=10):
        super(unit_gcn, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.Times = Times
        self.num_subset = A.shape[0]
        self.adaptive = adaptive
        if adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)

        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv1d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels))
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm1d(out_channels)
        self.SSA = MSA_Conv(out_channels, num_heads=8)
        self.lif1 = neuron.LIFNode(step_mode='m', backend='cupy')
        self.lif2 = neuron.LIFNode(step_mode='m', backend='cupy')

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm1d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def L2_norm(self, A):
        A_norm = torch.norm(A, 2, dim=1, keepdim=True) + 1e-4
        A = A / A_norm
        return A

    def forward(self, x):
        T, N, C, V = x.size()
        s = x.view(T * N, C, V)
        y = None
        if self.adaptive:
            A = self.PA
            A = self.L2_norm(A)
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            A1 = A[i]
            A2 = x.view(T * N, C, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(T * N, C, V))
            y = z + y if y is not None else z
        y = self.bn(y).reshape(T, N, self.out_c, V).contiguous()
        y = self.lif1(y)
        y = y + self.lif2(self.down(s).reshape(T, N, self.out_c, V).contiguous())
        y = self.SSA(y)
        return y


class TCN_GCN_unit(nn.Module):
    """1D-SGC + FSC stacked unit"""

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, Times=10, num_point=25):
        super(TCN_GCN_unit, self).__init__()
        self.Times = Times
        self.shortcut_lif = neuron.LIFNode(step_mode='m', backend='cupy')
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive, Times=self.Times)
        self.fa = SpikingAttention(in_channels=out_channels, out_channel=out_channels, num_point=num_point)

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = SpikingAttention(in_channels=in_channels, out_channel=out_channels, num_point=num_point)

    def forward(self, x):
        y = self.gcn1(x)
        y1 = self.fa(y)
        y2 = self.residual(x)
        return y + y1 + y2


# ============ Utilities ============
class NormalizeSkeleton(nn.Module):
    def __init__(self, num_features):
        super(NormalizeSkeleton, self).__init__()
        self.batch_norm = nn.BatchNorm1d(num_features)

    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(0, 2, 3, 1).contiguous().view(N * T, V * C)
        x = self.batch_norm(x)
        x = x.view(N, T, V, C)
        for c in range(C):
            coord = x[:, :, :, c]
            coord_min = coord.min(dim=2, keepdim=True)[0]
            coord_max = coord.max(dim=2, keepdim=True)[0]
            coord = (coord - coord_min) / (coord_max - coord_min + 1e-8)
            x[:, :, :, c] = coord
        x = x.permute(1, 0, 3, 2).contiguous()
        return x


# ============ Signal-SGN Model ============
class Model(nn.Module):
    """Signal-SGN: Full model with 1D-SGC, FSC, and MWTF"""

    def __init__(self, num_class=120, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0.3, adaptive=True, num_set=3, Times=4):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError("graph must be specified")
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = np.stack([np.eye(num_point)] * num_set, axis=0)
        self.Times = Times
        self.num_class = num_class
        self.num_point = num_point
        self.skeletonbn = NormalizeSkeleton(3 * num_point)
        self.base_channel = 64
        self.num_frames = 16
        self.num_person = num_person
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(3, 64, A, adaptive=adaptive, Times=self.Times, num_point=self.num_point)
        self.l2 = TCN_GCN_unit(64, 64, A, adaptive=adaptive, Times=self.Times, num_point=self.num_point)
        self.l5 = TCN_GCN_unit(64, 128, A, adaptive=adaptive, stride=2, Times=self.Times, num_point=self.num_point)
        self.l8 = TCN_GCN_unit(128, 256, A, adaptive=adaptive, stride=2, Times=self.Times, num_point=self.num_point)

        self.dwtfc = MultiWaveletTransform(
            num_frame=self.num_frames, num_point=self.num_point, num_class=self.num_class
        )
        # md: learnable beta balancing time-domain and frequency-domain features
        self.alpha = nn.Parameter(torch.tensor(0.42))
        self.lv = nn.Linear(256, 256)
        self.bnv = nn.BatchNorm1d(256)
        self.lvlif = neuron.IFNode(step_mode='m', backend='cupy')
        self.fc = nn.Linear(256, num_class)

        nn.init.normal_(self.lv.weight, 0, math.sqrt(2. / 256))
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x, label=None):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2)
        x = x.view(N * M, C, T, V)
        x = self.skeletonbn(x)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l5(x)
        x = self.l8(x)

        x = x.view(x.size(0), N * M, -1, V).permute(1, 2, 0, 3)
        x = x.view(N, M, x.size(1), x.size(2), x.size(3))
        x = x.mean(1)

        backbone_x = x  # X^(L) in md
        fused_output, fused_outputs_d, fused_outputs_x = self.dwtfc(backbone_x)

        fused_output = fused_output.mean(3)
        fused_output = fused_output.permute(1, 0, 2).contiguous()
        fused_output = self.lv(fused_output).permute(0, 2, 1).contiguous()
        fused_output = self.bnv(fused_output)
        fused_output = self.lvlif(fused_output)
        fused_output = fused_output.permute(2, 1, 0).contiguous()
        fused_output = fused_output.mean(2)

        wave_feat = self.drop_out(fused_output)  # hatX from MWTF
        # md: GAP(SN(X^(L))) + beta * hatX
        time_feat = backbone_x.mean(dim=3).mean(dim=2)  # [N, 256]
        out = self.fc(time_feat + self.alpha * wave_feat)

        return out, self.alpha
