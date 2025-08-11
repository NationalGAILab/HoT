import sys
import torch
import torch.nn as nn
import math
import numpy as np
from collections import OrderedDict
from functools import partial
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., st_mode='vanilla'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.mode = st_mode
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, seqlen=1):
        B, N, C = x.shape
        
        if self.mode == 'temporal':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = self.forward_temporal(q, k, v, seqlen=seqlen)
        elif self.mode == 'spatial':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = self.forward_spatial(q, k, v)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def forward_spatial(self, q, k, v):
        B, _, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1,2).reshape(B, N, C*self.num_heads)
        return x
        
    def forward_temporal(self, q, k, v, seqlen=8):
        B, _, N, C = q.shape
        qt = q.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)
        kt = k.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)
        vt = v.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)

        attn = (qt @ kt.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ vt
        x = x.permute(0, 3, 2, 1, 4).reshape(B, N, C*self.num_heads)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, st_mode='stage_st'):
        super().__init__()
        self.st_mode = st_mode
        self.norm1_s = norm_layer(dim)
        self.norm1_t = norm_layer(dim)
        self.attn_s = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, st_mode="spatial")
        self.attn_t = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, st_mode="temporal")
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2_s = norm_layer(dim)
        self.norm2_t = norm_layer(dim)
        self.mlp_s = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.mlp_t = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)

    def forward(self, x, seqlen=1):
        if self.st_mode=='stage_st':
            x = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen))
            x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))
            x = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen))
            x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
        elif self.st_mode=='stage_ts':
            x = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen))
            x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
            x = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen))
            x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))

        return x
    
class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        dim_feat = 256
        mlp_hidden_dim = 256 * 4
        dim_rep = 512
        depth = 5
        num_heads = 8
        maxlen = 243

        drop_path_rate = 0.0 
        drop_rate = 0.
        attn_drop_rate = 0.
        qkv_bias = True
        qk_scale = None
        att_fuse = True

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.joints_embed = nn.Linear(3, dim_feat)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks_st = nn.ModuleList([
            Block(
                dim=dim_feat, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, st_mode="stage_st")
            for i in range(depth)])
        
        self.blocks_ts = nn.ModuleList([
            Block(
                dim=dim_feat, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, st_mode="stage_ts")
            for i in range(depth)])

        self.norm = norm_layer(dim_feat)

        self.pre_logits = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(dim_feat, dim_rep)),
            ('act', nn.Tanh())
        ]))

        self.head = nn.Linear(dim_rep, 3)

        self.temp_embed = nn.Parameter(torch.zeros(1, maxlen, 1, dim_feat))
        self.pos_embed = nn.Parameter(torch.zeros(1, 17, dim_feat))
        trunc_normal_(self.temp_embed, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

        self.ts_attn = nn.ModuleList([nn.Linear(dim_feat*2, 2) for i in range(depth)])
        for i in range(depth):
            self.ts_attn[i].weight.data.fill_(0)
            self.ts_attn[i].bias.data.fill_(0.5)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):   
        B, F, J, C = x.shape

        x = rearrange(x, 'b f n c  -> (b f) n c')
        x = self.joints_embed(x)
        x = x + self.pos_embed

        x = rearrange(x, '(b f) n c -> b f n c', f=F)

        x = x + self.temp_embed
        x = rearrange(x, 'b f n c  -> (b f) n c')
        x = self.pos_drop(x)

        for idx, (blk_st, blk_ts) in enumerate(zip(self.blocks_st, self.blocks_ts)):
            x_st = blk_st(x, F)
            x_ts = blk_ts(x, F)

            att = self.ts_attn[idx]
            alpha = torch.cat([x_st, x_ts], dim=-1)
            alpha = att(alpha)
            alpha = alpha.softmax(dim=-1)
            x = x_st * alpha[:,:,0:1] + x_ts * alpha[:,:,1:2]

        x = self.norm(x)
        x = rearrange(x, '(b f) n c -> b f n c', f=F)
        
        x = self.pre_logits(x)
        x = self.head(x)

        return x


if __name__ == '__main__':
    import argparse
    import time

    args = argparse.ArgumentParser().parse_args()
    args.depth, args.dim_feat, args.mlp_ratio, args.dim_rep = 5, 256, 4, 512
    args.num_heads, args.maxlen = 8, 243

    input_2d = torch.rand(1, args.maxlen, 17, 3)

    with torch.no_grad():
        model = Model(args)
        model.eval()

        model_params = 0
        for parameter in model.parameters():
            model_params += parameter.numel()
        print('INFO: Trainable parameter count:', model_params/ 1000000)

        print(input_2d.shape, 1)
        output = model(input_2d)
        print(output.shape, 2)

    from thop import profile
    from thop import clever_format
    macs, params = profile(model, inputs=(input_2d, ))
    print('macs: ', macs/1000000, 'params: ', params/1000000)
    macs, params = clever_format([macs*2, params], "%.3f")
    print(macs, params)    


