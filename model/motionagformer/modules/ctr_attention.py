import torch
from torch import nn
from torch.autograd import Variable

CONNECTIONS = {10: [9], 9: [8, 10], 8: [7, 9], 14: [15, 8], 15: [16, 14], 11: [12, 8], 12: [13, 11],
               7: [0, 8], 0: [1, 7], 1: [2, 0], 2: [3, 1], 4: [5, 0], 5: [6, 4], 16: [15], 13: [12], 3: [2], 6: [5]}


class CTRAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 downsample_ratio=2, adaptive=True, n_frames=243, temporal_connection_length=1, mode='spatial',
                 use_self_similarity=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.inter_channel = dim // downsample_ratio

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.inter_channel, dim)
        self.mode = mode
        self.adaptive = adaptive
        self.use_self_similarity = use_self_similarity
        self.n_frames = n_frames
        self.temporal_connection_length = temporal_connection_length
        self.qkv = nn.Linear(dim, self.inter_channel * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.use_self_similarity:
            self.fc_mask_attn = nn.Linear(in_features=2, out_features=1, bias=False)
        else:
            self.alpha = nn.Parameter(torch.zeros(1))
            if self.mode == "spatial":
                self.shared_attn = self._init_spatial_shared_attn()
            elif self.mode == "temporal":
                self.shared_attn = self._init_temporal_shared_attn()

    def _init_spatial_shared_attn(self):
        num_joints = len(CONNECTIONS)
        attn = torch.zeros((num_joints, num_joints))

        for i in range(num_joints):
            connected_nodes = CONNECTIONS[i]
            attn[i, i] = 1
            for j in connected_nodes:
                attn[i, j] = 1

        if self.adaptive:
            return nn.Parameter(attn)
        else:
            return Variable(attn, requires_grad=False)

    def _init_temporal_shared_attn(self):
        attn = torch.zeros((self.n_frames, self.n_frames))

        for i in range(self.n_frames):
            for j in range(self.temporal_connection_length + 1):
                try:
                    attn[i, i + j] = 1
                except IndexError:  # next j frame does not exist
                    pass
                try:
                    attn[i, i - j] = 1
                except IndexError:  # previous j frame does not exist
                    pass

        if self.adaptive:
            return nn.Parameter(attn)
        else:
            return Variable(attn, requires_grad=False)

    def forward(self, x):
        B, T, J, C = x.shape

        qkv = self.qkv(x).reshape(B, T, J, 3, self.num_heads,
                                  self.inter_channel // self.num_heads).permute(3, 0, 4, 1, 2, 5)  # (3, B, H, T, J, C)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.mode == 'temporal':
            xx = None
            if self.use_self_similarity:
                x_ = x.transpose(1, 2)  # (B, J, T, C)
                x_T = x_.transpose(2, 3)  # (B, J, C, T)
                xx = x_ @ x_T  # (B, J, T, T)
            x = self.forward_temporal(q, k, v, xx)
        elif self.mode == 'spatial':
            xx = None
            if self.use_self_similarity:
                xT = x.transpose(2, 3)  # (B, T, C, J)
                xx = x @ xT
            x = self.forward_spatial(q, k, v, xx)
        else:
            raise NotImplementedError(self.mode)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_spatial(self, q, k, v, xx):
        B, H, T, J, C = q.shape
        channel_attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, J, J)
        channel_attn = channel_attn.softmax(dim=-1)
        channel_attn = self.attn_drop(channel_attn)

        if xx is None:
            shared_attn = self.shared_attn[None, None, None, ...]  # (1, 1, 1, J, J)
            if not self.adaptive:
                shared_attn = self._change_shared_attn_device(channel_attn, shared_attn)
            attn = shared_attn + self.alpha * channel_attn
        else:
            xx = xx.repeat(H, 1, 1, 1, 1).transpose(0, 1)  # (B, T, J, J) -> (B, H, T, J, J)
            attn = torch.cat((xx[..., None], channel_attn[..., None]), dim=-1)  # (B, H, T, J, J, 2)
            attn = self.fc_mask_attn(attn).squeeze(-1).softmax(dim=-1)

        x = attn @ v  # (B, H, T, J, C)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, T, J, C * self.num_heads)
        return x  # (B, T, J, C)

    def _change_shared_attn_device(self, channel_attn, shared_attn):
        dev = channel_attn.get_device()
        if dev >= 0:
            shared_attn = shared_attn.to(dev)
        return shared_attn

    def forward_temporal(self, q, k, v, xx):
        B, H, T, J, C = q.shape
        qt = q.transpose(2, 3)  # (B, H, J, T, C)
        kt = k.transpose(2, 3)  # (B, H, J, T, C)
        vt = v.transpose(2, 3)  # (B, H, J, T, C)

        channel_attn = (qt @ kt.transpose(-2, -1)) * self.scale  # (B, H, J, T, T)
        channel_attn = channel_attn.softmax(dim=-1)
        channel_attn = self.attn_drop(channel_attn)

        if xx is None:
            shared_attn = self.shared_attn[None, None, None, ...]  # (1, 1, 1, T, T)
            if not self.adaptive:
                shared_attn = self._change_shared_attn_device(channel_attn, shared_attn)
            attn = shared_attn + self.alpha * channel_attn
        else:
            xx = xx.repeat(H, 1, 1, 1, 1).transpose(0, 1)  # (B, J, T, T) -> (B, H, J, T, T)
            attn = torch.cat((xx[..., None], channel_attn[..., None]), dim=-1)  # (B, H, J, T, T, 2)
            attn = self.fc_mask_attn(attn).squeeze(-1).softmax(dim=-1)  # (B, H, J, T, T)

        x = attn @ vt  # (B, H, J, T, C)
        x = x.permute(0, 3, 2, 1, 4).reshape(B, T, J, C * self.num_heads)
        return x  # (B, T, J, C)
