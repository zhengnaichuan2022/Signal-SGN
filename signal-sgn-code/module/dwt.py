import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import List, Tuple
import math
from functools import partial
from einops import rearrange, reduce, repeat
from torch import nn, einsum, diagonal
from math import log2, ceil
from scipy.special import eval_legendre
from sympy import Poly, legendre, Symbol, chebyshevt
from spikingjelly.activation_based import neuron, layer, functional
def legendreDer(k, x):
    def _legendre(k, x):
        return (2*k+1) * eval_legendre(k, x)
    out = 0
    for i in np.arange(k-1,-1,-2):
        out += _legendre(i, x)
    return out


def phi_(phi_c, x, lb = 0, ub = 1):
    mask = np.logical_or(x<lb, x>ub) * 1.0
    return np.polynomial.polynomial.Polynomial(phi_c)(x) * (1-mask)


def get_phi_psi(k, base):
    
    x = Symbol('x')
    phi_coeff = np.zeros((k,k))
    phi_2x_coeff = np.zeros((k,k))
    if base == 'legendre':
        for ki in range(k):
            coeff_ = Poly(legendre(ki, 2*x-1), x).all_coeffs()
            phi_coeff[ki,:ki+1] = np.flip(np.sqrt(2*ki+1) * np.array(coeff_).astype(np.float64))
            coeff_ = Poly(legendre(ki, 4*x-1), x).all_coeffs()
            phi_2x_coeff[ki,:ki+1] = np.flip(np.sqrt(2) * np.sqrt(2*ki+1) * np.array(coeff_).astype(np.float64))
        
        psi1_coeff = np.zeros((k, k))
        psi2_coeff = np.zeros((k, k))
        for ki in range(k):
            psi1_coeff[ki,:] = phi_2x_coeff[ki,:]
            for i in range(k):
                a = phi_2x_coeff[ki,:ki+1]
                b = phi_coeff[i, :i+1]
                prod_ = np.convolve(a, b)
                prod_[np.abs(prod_)<1e-8] = 0
                proj_ = (prod_ * 1/(np.arange(len(prod_))+1) * np.power(0.5, 1+np.arange(len(prod_)))).sum()
                psi1_coeff[ki,:] -= proj_ * phi_coeff[i,:]
                psi2_coeff[ki,:] -= proj_ * phi_coeff[i,:]
            for j in range(ki):
                a = phi_2x_coeff[ki,:ki+1]
                b = psi1_coeff[j, :]
                prod_ = np.convolve(a, b)
                prod_[np.abs(prod_)<1e-8] = 0
                proj_ = (prod_ * 1/(np.arange(len(prod_))+1) * np.power(0.5, 1+np.arange(len(prod_)))).sum()
                psi1_coeff[ki,:] -= proj_ * psi1_coeff[j,:]
                psi2_coeff[ki,:] -= proj_ * psi2_coeff[j,:]

            a = psi1_coeff[ki,:]
            prod_ = np.convolve(a, a)
            prod_[np.abs(prod_)<1e-8] = 0
            norm1 = (prod_ * 1/(np.arange(len(prod_))+1) * np.power(0.5, 1+np.arange(len(prod_)))).sum()

            a = psi2_coeff[ki,:]
            prod_ = np.convolve(a, a)
            prod_[np.abs(prod_)<1e-8] = 0
            norm2 = (prod_ * 1/(np.arange(len(prod_))+1) * (1-np.power(0.5, 1+np.arange(len(prod_))))).sum()
            norm_ = np.sqrt(norm1 + norm2)
            psi1_coeff[ki,:] /= norm_
            psi2_coeff[ki,:] /= norm_
            psi1_coeff[np.abs(psi1_coeff)<1e-8] = 0
            psi2_coeff[np.abs(psi2_coeff)<1e-8] = 0

        phi = [np.poly1d(np.flip(phi_coeff[i,:])) for i in range(k)]
        psi1 = [np.poly1d(np.flip(psi1_coeff[i,:])) for i in range(k)]
        psi2 = [np.poly1d(np.flip(psi2_coeff[i,:])) for i in range(k)]
    
    elif base == 'chebyshev':
        for ki in range(k):
            if ki == 0:
                phi_coeff[ki,:ki+1] = np.sqrt(2/np.pi)
                phi_2x_coeff[ki,:ki+1] = np.sqrt(2/np.pi) * np.sqrt(2)
            else:
                coeff_ = Poly(chebyshevt(ki, 2*x-1), x).all_coeffs()
                phi_coeff[ki,:ki+1] = np.flip(2/np.sqrt(np.pi) * np.array(coeff_).astype(np.float64))
                coeff_ = Poly(chebyshevt(ki, 4*x-1), x).all_coeffs()
                phi_2x_coeff[ki,:ki+1] = np.flip(np.sqrt(2) * 2 / np.sqrt(np.pi) * np.array(coeff_).astype(np.float64))
                
        phi = [partial(phi_, phi_coeff[i,:]) for i in range(k)]
        
        x = Symbol('x')
        kUse = 2*k
        roots = Poly(chebyshevt(kUse, 2*x-1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        # x_m[x_m==0.5] = 0.5 + 1e-8 # add small noise to avoid the case of 0.5 belonging to both phi(2x) and phi(2x-1)
        # not needed for our purpose here, we use even k always to avoid
        wm = np.pi / kUse / 2
        
        psi1_coeff = np.zeros((k, k))
        psi2_coeff = np.zeros((k, k))

        psi1 = [[] for _ in range(k)]
        psi2 = [[] for _ in range(k)]

        for ki in range(k):
            psi1_coeff[ki,:] = phi_2x_coeff[ki,:]
            for i in range(k):
                proj_ = (wm * phi[i](x_m) * np.sqrt(2)* phi[ki](2*x_m)).sum()
                psi1_coeff[ki,:] -= proj_ * phi_coeff[i,:]
                psi2_coeff[ki,:] -= proj_ * phi_coeff[i,:]

            for j in range(ki):
                proj_ = (wm * psi1[j](x_m) * np.sqrt(2) * phi[ki](2*x_m)).sum()        
                psi1_coeff[ki,:] -= proj_ * psi1_coeff[j,:]
                psi2_coeff[ki,:] -= proj_ * psi2_coeff[j,:]

            psi1[ki] = partial(phi_, psi1_coeff[ki,:], lb = 0, ub = 0.5)
            psi2[ki] = partial(phi_, psi2_coeff[ki,:], lb = 0.5, ub = 1)

            norm1 = (wm * psi1[ki](x_m) * psi1[ki](x_m)).sum()
            norm2 = (wm * psi2[ki](x_m) * psi2[ki](x_m)).sum()

            norm_ = np.sqrt(norm1 + norm2)
            psi1_coeff[ki,:] /= norm_
            psi2_coeff[ki,:] /= norm_
            psi1_coeff[np.abs(psi1_coeff)<1e-8] = 0
            psi2_coeff[np.abs(psi2_coeff)<1e-8] = 0

            psi1[ki] = partial(phi_, psi1_coeff[ki,:], lb = 0, ub = 0.5+1e-16)
            psi2[ki] = partial(phi_, psi2_coeff[ki,:], lb = 0.5+1e-16, ub = 1)
        
    return phi, psi1, psi2


def get_filter(base, k):
    
    def psi(psi1, psi2, i, inp):
        mask = (inp<=0.5) * 1.0
        return psi1[i](inp) * mask + psi2[i](inp) * (1-mask)
    
    if base not in ['legendre', 'chebyshev']:
        raise Exception('Base not supported')
    
    x = Symbol('x')
    H0 = np.zeros((k,k))
    H1 = np.zeros((k,k))
    G0 = np.zeros((k,k))
    G1 = np.zeros((k,k))
    PHI0 = np.zeros((k,k))
    PHI1 = np.zeros((k,k))
    phi, psi1, psi2 = get_phi_psi(k, base)
    if base == 'legendre':
        roots = Poly(legendre(k, 2*x-1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        wm = 1/k/legendreDer(k,2*x_m-1)/eval_legendre(k-1,2*x_m-1)
        
        for ki in range(k):
            for kpi in range(k):
                H0[ki, kpi] = 1/np.sqrt(2) * (wm * phi[ki](x_m/2) * phi[kpi](x_m)).sum()
                G0[ki, kpi] = 1/np.sqrt(2) * (wm * psi(psi1, psi2, ki, x_m/2) * phi[kpi](x_m)).sum()
                H1[ki, kpi] = 1/np.sqrt(2) * (wm * phi[ki]((x_m+1)/2) * phi[kpi](x_m)).sum()
                G1[ki, kpi] = 1/np.sqrt(2) * (wm * psi(psi1, psi2, ki, (x_m+1)/2) * phi[kpi](x_m)).sum()
                
        PHI0 = np.eye(k)
        PHI1 = np.eye(k)
                
    elif base == 'chebyshev':
        x = Symbol('x')
        kUse = 2*k
        roots = Poly(chebyshevt(kUse, 2*x-1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        # x_m[x_m==0.5] = 0.5 + 1e-8 # add small noise to avoid the case of 0.5 belonging to both phi(2x) and phi(2x-1)
        # not needed for our purpose here, we use even k always to avoid
        wm = np.pi / kUse / 2

        for ki in range(k):
            for kpi in range(k):
                H0[ki, kpi] = 1/np.sqrt(2) * (wm * phi[ki](x_m/2) * phi[kpi](x_m)).sum()
                G0[ki, kpi] = 1/np.sqrt(2) * (wm * psi(psi1, psi2, ki, x_m/2) * phi[kpi](x_m)).sum()
                H1[ki, kpi] = 1/np.sqrt(2) * (wm * phi[ki]((x_m+1)/2) * phi[kpi](x_m)).sum()
                G1[ki, kpi] = 1/np.sqrt(2) * (wm * psi(psi1, psi2, ki, (x_m+1)/2) * phi[kpi](x_m)).sum()

                PHI0[ki, kpi] = (wm * phi[ki](2*x_m) * phi[kpi](2*x_m)).sum() * 2
                PHI1[ki, kpi] = (wm * phi[ki](2*x_m-1) * phi[kpi](2*x_m-1)).sum() * 2
                
        PHI0[np.abs(PHI0)<1e-8] = 0
        PHI1[np.abs(PHI1)<1e-8] = 0

    H0[np.abs(H0)<1e-8] = 0
    H1[np.abs(H1)<1e-8] = 0
    G0[np.abs(G0)<1e-8] = 0
    G1[np.abs(G1)<1e-8] = 0
        
    return H0, H1, G0, G1, PHI0, PHI1

# Till EoF
# taken from FNO paper:
# https://github.com/zongyi-li/fourier_neural_operator

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x
    
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
    



class MultiWaveletTransform(nn.Module):
    def __init__(self, k=8, alpha=16, c=3,
                 nCZ=1, L=0, J=3, base='legendre', attention_dropout=0.1, num_frame=64, num_point=25, num_class=60):
        super(MultiWaveletTransform,self).__init__()
        self.k = k
        self.c = c
        self.L = L
        self.J = J
        self.ich = c
        self.nCZ = nCZ
        self.down1 = nn.Conv2d(256, 64, kernel_size=3, padding=1, groups=16)
        self.down2 = nn.Sequential (nn.Conv2d(64, k, kernel_size=3, padding=1, groups=1),
                                    nn.BatchNorm2d(k))
        self.Lk0 =nn.Linear(k, k)
        # self.downbn = nn.BatchNorm1d(c*k,c* k)
        # self.Lk1 = nn.Linear(c * k, self.ich)
        # MWTF core: recursive DWT + spiking cross-attention fusion
        # md suggests best decomposition level J=3, so keep J explicit.
        self.MWT_CZ = MWT_CZ1d(k=k, alpha=alpha, L=L, c=c, base=base, out_channel=256, J=J)
        # self.classfc1 = classfication(in_channel= 256,out_channel=256,num_frame=num_frame
        #                               ,num_point=num_point,num_class=num_class) 

    def forward(self,x):
        # x = x.permute(1,2,0,3).contiguous()
        # print(x.size())
        N,C,T,V = x.size()
        x = self.down1(x)
        x = self.down2(x)       

        x = x.view(N,-1,V,T).permute(0,3,2,1).contiguous()
        # 
        x = x.flatten(0,1)
        x = self.Lk0(x)
        # x = self.downbn(x)         
        x = x.view(N,T,self.k,-1)


        x,fused_outputs_d,fused_outputs_x= self.MWT_CZ(x)
        # x =x.mean(3).mean(2)
 
        # x = self.classfc1(x)

        return x,fused_outputs_d,fused_outputs_x

class CrossAttention(nn.Module):
    def __init__(
        self,
        in_dim,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        mode="direct_xor",
        spike_mode="lif",
        dvs=False,
        layer=0,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.dvs = dvs
        self.num_heads = num_heads

        self.scale = 0.125
        self.q_conv = nn.Conv1d(in_dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)

        self.q_lif = neuron.IFNode(step_mode='m',backend='cupy')


        self.k_conv = nn.Conv1d(in_dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif =neuron.IFNode(step_mode='m',backend='cupy')

        self.v_conv = nn.Conv1d(in_dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif =neuron.IFNode(step_mode='m',backend='cupy')
        self.attn_lif = neuron.IFNode(step_mode='m',backend='cupy',v_threshold=0.5)


        self.talking_heads = nn.Conv1d(
            num_heads, num_heads, kernel_size=1, stride=1, bias=False
        )
        self.talking_heads_lif = neuron.IFNode(step_mode='m',backend='cupy',v_threshold=0.5)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = neuron.LIFNode(step_mode='m',backend='cupy')
        self.shortcut_lif1 = neuron.IFNode(step_mode='m',backend='cupy',v_threshold=0.5)
        self.shortcut_lif2 = neuron.IFNode(step_mode='m',backend='cupy',v_threshold=0.5)
        self.mode = mode
        self.layer = layer

    def forward(self, x, y):
        x = x.permute(1,0,2,3).contiguous()
        y = y.permute(1,0,2,3).contiguous()
        C2 = self.dim
        T,N,C1,V = x.shape
        identity = x
        x = self.shortcut_lif1(x)
        x_for_qk = x.flatten(0, 1)
        y = self.shortcut_lif2(y)
        y_for_qk = y.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qk)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, N, C2, V).contiguous()
        q_conv_out = self.q_lif(q_conv_out)

        q = (
            q_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, N, V, self.num_heads, C2 // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k_conv_out = self.k_conv(x_for_qk)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, N, C2, V).contiguous()
        k_conv_out = self.k_lif(k_conv_out)

        k = (
            k_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, N, V, self.num_heads, C2// self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v_conv_out = self.v_conv(y_for_qk)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, N, C2, V).contiguous()
        v_conv_out = self.v_lif(v_conv_out)

        v = (
            v_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, N, V, self.num_heads, C2 // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )  # T B head N C//h

        kv = k.mul(v)

        kv = kv.sum(dim=-2, keepdim=True)
        kv = self.talking_heads_lif(kv)

        x = q.mul(kv)

        x = x.transpose(3, 4).reshape(T, N, C2, V).contiguous()
        x = (
            self.proj_bn(self.proj_conv(x.flatten(0, 1)))
            .reshape(T,N, C2,V)
            .contiguous()
        )
        x = self.proj_lif(x)
        x = x.permute(1,0,2,3).contiguous()
        # y = y.permute(1,0,2,3).contiguous()
        return x

# class CrossAttention(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super(CrossAttention, self).__init__()
#         self.out_dim = out_dim
#         self.query = nn.Conv2d(in_dim, out_dim, kernel_size=1)
#         self.q_bn = nn.BatchNorm2d(out_dim,out_dim)
#         self.key = nn.Conv2d(in_dim, out_dim, kernel_size=1)
#         self.k_bn = nn.BatchNorm2d(out_dim,out_dim)
#         self.value = nn.Conv2d(in_dim, out_dim, kernel_size=1)
#         self.v_bn = nn.BatchNorm2d(out_dim,out_dim)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x1, x2):

#         N, T, C, V = x1.shape
        
#         # Apply convolution and reshape for query, key, and value
#         query = self.query(x1.permute(0, 2, 1, 3))
#         query= self.q_bn(query).view(N,self.out_dim, -1)
#         key = self.key(x2.permute(0, 2, 1, 3))
#         key = self.k_bn(key).view(N, self.out_dim, -1)
#         value = self.value(x2.permute(0, 2, 1, 3))
#         value = self.v_bn(value).view(N, self.out_dim, -1)
#         # Compute attention
#         attention = self.softmax(torch.bmm(query.permute(0, 2, 1), key))
        
#         # Apply attention to value and reshape output
#         out = torch.bmm(value, attention.permute(0, 2, 1)).view(N, self.out_dim, T, V)
        
#         return out


class MWT_CZ1d(nn.Module):
    def __init__(self,
                 k=3, alpha=64,
                 L=0, c=1,
                 J=3,
                 base='legendre',
                 initializer=None,
                 out_channel=256,
                 **kwargs):
        super(MWT_CZ1d, self).__init__()
        self.shortcut_lif2 = neuron.IFNode(step_mode='m',backend='cupy',v_threshold=0.5)
        self.k = k
        self.L = L
        self.J = J
        # md: learnable scaling factor for cross-attention fusion outputs
        self.s = nn.Parameter(torch.tensor(1.0))
        self.cross_attention= CrossAttention(k, out_channel)
        # self.fc = classfication()
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1

        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0
        self.max_item = 3
        self.register_buffer('ec_s', torch.Tensor(
            np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(
            np.concatenate((G0.T, G1.T), axis=0)))

        self.register_buffer('rc_e', torch.Tensor(
            np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer('rc_o', torch.Tensor(
            np.concatenate((H1r, G1r), axis=0)))
    def forward(self,x):

        N,T,C,V = x.shape  # (B, N, k)
        target_T = T  # md: upsample each coefficient back to original temporal resolution
        ns = math.floor(np.log2(T))
        nl = pow(2, math.ceil(np.log2(T)))
        extra_x = x[:, 0:nl - T, :, :]
        x = torch.cat([x, extra_x], 1)
        d_list = []
        x_list = []
        # md uses J levels; keep it explicit and cap by feasible levels.
        num_levels = max(0, min(self.J, ns - self.L))
        for i in range(num_levels):
            d, x = self.wavelet_transform(x)
            d_list.append(d)
            x_list.append(x)
                
        fused_outputs_d = []
        fused_outputs_x = []
        fused_output = None
        for d, x in zip(d_list, x_list):
            # out_d = self.cross_attention(d, x)
            # out_x = self.cross_attention(x, d)
            # out_d = self.upsample_to_match(out_d, self.TI)
            # out_x = self.upsample_to_match(out_x, self.TI)
            # fused_outputs_d.append(out_d)
            # fused_outputs_x.append(out_x)
            d = self.upsample_to_match(d, target_T)
            x = self.upsample_to_match(x, target_T)
            out_d = self.cross_attention(d, x)
            out_x = self.cross_attention(x, d)
            out_d = out_d * self.s
            out_x = out_x * self.s
            fused_outputs_d.append(out_d)
            fused_outputs_x.append(out_x)
            if fused_output is not None:
                fused_output = fused_output + out_d + out_x
            else:
                fused_output = out_d + out_x
        
        fused_output = fused_output.permute(1,0,2,3).contiguous()
        fused_output = self.shortcut_lif2 (fused_output)
        fused_output = fused_output.permute(1,0,2,3).contiguous()
        # fused_output = self.fc(fused_output)
        return fused_output,fused_outputs_d,fused_outputs_x
    
    def upsample_to_match(self, x, target):
        N, T, C, V = x.shape
        x = x.view(N,C,V,T)
        x = x.flatten(1,2)
 
        interpolated_features = F.interpolate(x, size=target, mode='linear', align_corners=True)

        return interpolated_features.view(N,interpolated_features.size(2), C, V)

    def wavelet_transform(self, x):

        xa = torch.cat([x[:, ::2, :, :],
                        x[:, 1::2, :, :],
                        ], -2)
        xa=xa.permute(0,1,3,2).contiguous()
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        d=d.permute(0,1,3,2).contiguous()
        s = s.permute(0,1,3,2).contiguous()

        return d, s

class classfication(nn.Module):
    def __init__(self, in_channel=256,out_channel=256,num_frame=64,num_point=25,num_class=60):
        super(classfication, self).__init__()
        self.c = out_channel
        self.decoder=nn.Linear(in_channel,out_channel)
    def forward(self,x):
        x =x.mean(3).mean(2)

        x =self.decoder(x)

        return x