import sys
from collections import OrderedDict
import torch
from torch import nn
from timm.models.layers import DropPath
from model.motionagformer.modules.attention import Attention
from model.motionagformer.modules.graph import GCN
from model.motionagformer.modules.mlp import MLP
from model.motionagformer.modules.tcn import MultiScaleTCN
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]

    return new_points


def cluster_dpc_knn(x, cluster_num, k, token_mask=None):
    with torch.no_grad():
        B, N, C = x.shape

        dist_matrix = torch.cdist(x, x) / (C ** 0.5)

        if token_mask is not None:
            token_mask = token_mask > 0
            dist_matrix = dist_matrix * token_mask[:, None, :] + (dist_matrix.max() + 1) * (~token_mask[:, None, :])

        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)

        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6

        if token_mask is not None:
            density = density * token_mask

        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        score = dist * density
        
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        dist_matrix = index_points(dist_matrix, index_down)

        idx_cluster = dist_matrix.argmin(dim=1)

        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return index_down, idx_cluster


class AGFormerBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, qkv_bias=False, qk_scale=None, use_layer_scale=True, layer_scale_init_value=1e-5,
                 mode='spatial', mixer_type="attention", use_temporal_similarity=True,
                 temporal_connection_len=1, neighbour_num=4, n_frames=243):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if mixer_type == 'attention':
            self.mixer = Attention(dim, dim, num_heads, qkv_bias, qk_scale, attn_drop,
                                   proj_drop=drop, mode=mode)
        elif mixer_type == 'graph':
            self.mixer = GCN(dim, dim,
                             num_nodes=17 if mode == 'spatial' else n_frames,
                             neighbour_num=neighbour_num,
                             mode=mode,
                             use_temporal_similarity=use_temporal_similarity,
                             temporal_connection_len=temporal_connection_len)
        elif mixer_type == "ms-tcn":
            self.mixer = MultiScaleTCN(in_channels=dim, out_channels=dim)
        else:
            raise NotImplementedError("AGFormer mixer_type is either attention or graph")
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(0).unsqueeze(0)
                * self.mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., length=27):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.linear_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_1, x_2, x_3):
        B, N, C = x_1.shape
        B, N_1, C = x_3.shape

        q = self.linear_q(x_1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.linear_k(x_2).reshape(B, N_1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.linear_v(x_3).reshape(B, N_1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MotionAGFormerBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, use_layer_scale=True, qkv_bias=False, qk_scale=None, layer_scale_init_value=1e-5,
                 use_adaptive_fusion=True, hierarchical=False, use_temporal_similarity=True,
                 temporal_connection_len=1, use_tcn=False, graph_only=False, neighbour_num=4, n_frames=243):
        super().__init__()
        self.hierarchical = hierarchical
        dim = dim // 2 if hierarchical else dim

        self.att_spatial = AGFormerBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                         qk_scale, use_layer_scale, layer_scale_init_value,
                                         mode='spatial', mixer_type="attention",
                                         use_temporal_similarity=use_temporal_similarity,
                                         neighbour_num=neighbour_num,
                                         n_frames=n_frames)
        self.att_temporal = AGFormerBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                          qk_scale, use_layer_scale, layer_scale_init_value,
                                          mode='temporal', mixer_type="attention",
                                          use_temporal_similarity=use_temporal_similarity,
                                          neighbour_num=neighbour_num,
                                          n_frames=n_frames)

        if graph_only:
            self.graph_spatial = GCN(dim, dim,
                                     num_nodes=17,
                                     mode='spatial')
            if use_tcn:
                self.graph_temporal = MultiScaleTCN(in_channels=dim, out_channels=dim)
            else:
                self.graph_temporal = GCN(dim, dim,
                                          num_nodes=n_frames,
                                          neighbour_num=neighbour_num,
                                          mode='temporal',
                                          use_temporal_similarity=use_temporal_similarity,
                                          temporal_connection_len=temporal_connection_len)
        else:
            self.graph_spatial = AGFormerBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads,
                                               qkv_bias,
                                               qk_scale, use_layer_scale, layer_scale_init_value,
                                               mode='spatial', mixer_type="graph",
                                               use_temporal_similarity=use_temporal_similarity,
                                               temporal_connection_len=temporal_connection_len,
                                               neighbour_num=neighbour_num,
                                               n_frames=n_frames)
            self.graph_temporal = AGFormerBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads,
                                                qkv_bias,
                                                qk_scale, use_layer_scale, layer_scale_init_value,
                                                mode='temporal', mixer_type="ms-tcn" if use_tcn else 'graph',
                                                use_temporal_similarity=use_temporal_similarity,
                                                temporal_connection_len=temporal_connection_len,
                                                neighbour_num=neighbour_num,
                                                n_frames=n_frames)

        self.use_adaptive_fusion = use_adaptive_fusion
        if self.use_adaptive_fusion:
            self.fusion = nn.Linear(dim * 2, 2)
            self._init_fusion()

    def _init_fusion(self):
        self.fusion.weight.data.fill_(0)
        self.fusion.bias.data.fill_(0.5)

    def forward(self, x):
        if self.hierarchical:
            B, T, J, C = x.shape
            x_attn, x_graph = x[..., :C // 2], x[..., C // 2:]

            x_attn = self.att_temporal(self.att_spatial(x_attn))
            x_graph = self.graph_temporal(self.graph_spatial(x_graph + x_attn))
        else:
            x_attn = self.att_temporal(self.att_spatial(x))
            x_graph = self.graph_temporal(self.graph_spatial(x))

        if self.hierarchical:
            x = torch.cat((x_attn, x_graph), dim=-1)
        elif self.use_adaptive_fusion:
            alpha = torch.cat((x_attn, x_graph), dim=-1)
            alpha = self.fusion(alpha)
            alpha = alpha.softmax(dim=-1)
            x = x_attn * alpha[..., 0:1] + x_graph * alpha[..., 1:2]
        else:
            x = (x_attn + x_graph) * 0.5

        return x


def create_layers(dim, n_layers, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop_rate=0., drop_path_rate=0.,
                  num_heads=8, use_layer_scale=True, qkv_bias=False, qkv_scale=None, layer_scale_init_value=1e-5,
                  use_adaptive_fusion=True, hierarchical=False, use_temporal_similarity=True,
                  temporal_connection_len=1, use_tcn=False, graph_only=False, neighbour_num=4, n_frames_all=243):
    layers = []
    for i in range(n_layers):
        layers.append(MotionAGFormerBlock(dim=dim,
                                          mlp_ratio=mlp_ratio,
                                          act_layer=act_layer,
                                          attn_drop=attn_drop,
                                          drop=drop_rate,
                                          drop_path=drop_path_rate,
                                          num_heads=num_heads,
                                          use_layer_scale=use_layer_scale,
                                          layer_scale_init_value=layer_scale_init_value,
                                          qkv_bias=qkv_bias,
                                          qk_scale=qkv_scale,
                                          use_adaptive_fusion=use_adaptive_fusion,
                                          hierarchical=hierarchical,
                                          use_temporal_similarity=use_temporal_similarity,
                                          temporal_connection_len=temporal_connection_len,
                                          use_tcn=use_tcn,
                                          graph_only=graph_only,
                                          neighbour_num=neighbour_num,
                                          n_frames=n_frames_all[i]))
    layers = nn.Sequential(*layers)

    return layers


class Model(nn.Module):
    def __init__(self, args, n_layers=12, dim_in=3, dim_feat=64, dim_rep=512, dim_out=3, mlp_ratio=4, act_layer=nn.GELU, attn_drop=0.,
                 drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5, use_adaptive_fusion=True,
                 num_heads=4, qkv_bias=False, qkv_scale=None, hierarchical=False, num_joints=17,
                 use_temporal_similarity=True, temporal_connection_len=1, use_tcn=False, graph_only=False,
                 neighbour_num=4, n_frames=243):
        super().__init__()

        n_frames = args.maxlen

        n_layers, dim_feat = 16, 128

        self.token_num = 81
        self.layer_index = 7

        n_frames_all = [243] * n_layers
        for i in range(self.layer_index, n_layers):
                n_frames_all[i] = self.token_num

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pos_embed_token = nn.Parameter(torch.zeros(1, self.token_num, dim_feat))

        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        self.norm = nn.LayerNorm(dim_feat)

        self.layers = create_layers(dim=dim_feat,
                                    n_layers=n_layers,
                                    mlp_ratio=mlp_ratio,
                                    act_layer=act_layer,
                                    attn_drop=attn_drop,
                                    drop_rate=drop,
                                    drop_path_rate=drop_path,
                                    num_heads=num_heads,
                                    use_layer_scale=use_layer_scale,
                                    qkv_bias=qkv_bias,
                                    qkv_scale=qkv_scale,
                                    layer_scale_init_value=layer_scale_init_value,
                                    use_adaptive_fusion=use_adaptive_fusion,
                                    hierarchical=hierarchical,
                                    use_temporal_similarity=use_temporal_similarity,
                                    temporal_connection_len=temporal_connection_len,
                                    use_tcn=use_tcn,
                                    graph_only=graph_only,
                                    neighbour_num=neighbour_num,
                                    n_frames_all=n_frames_all)

        ## cross
        self.x_token = nn.Parameter(torch.zeros(1, args.maxlen, dim_feat))
        self.cross_attention = Cross_Attention(dim_feat, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qkv_scale, attn_drop=attn_drop, proj_drop=drop)

        self.rep_logit = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(dim_feat, dim_rep)),
            ('act', nn.Tanh())
        ]))

        self.head = nn.Linear(dim_rep, dim_out)

    def forward(self, x):
        B, F, N, C = x.shape

        x = self.joints_embed(x)
        x = x + self.pos_embed

        for idx, layer in enumerate(self.layers):
            ##-----------------Clusteing-----------------##
            if idx == self.layer_index:
                x_knn = rearrange(x, 'b f n c -> b (f c) n')
                x_knn = self.pool(x_knn)
                x_knn = rearrange(x_knn, 'b (f c) 1 -> b f c', f=F)

                index, idx_cluster = cluster_dpc_knn(x_knn, self.token_num, 2)
                index, _ = torch.sort(index)
                batch_ind = torch.arange(B, device=x.device).unsqueeze(-1)
                x = x[batch_ind, index]

                B, F, N, C = x.shape

                x = rearrange(x, 'b f n c -> (b n) f c')
                x += self.pos_embed_token
                x = rearrange(x, '(b n) f c -> b f n c', b=B)
            ##-----------------Clusteing-----------------##

            x = layer(x)

        x = self.norm(x)

        x = rearrange(x, 'b f n c -> (b n) f c', b=B)
        x_token = repeat(self.x_token, '() f c -> b f c', b = B*N)
        x = x_token + self.cross_attention(x_token, x, x)
        x = rearrange(x, '(b n) f c -> b f n c', n=N)

        x = self.rep_logit(x)

        x = self.head(x)

        return x

