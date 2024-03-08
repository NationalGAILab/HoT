import torch
import torch.nn as nn
from einops import rearrange
from model.mhformer.module.trans import Transformer as Transformer_encoder
from model.mhformer.module.trans_hypothesis_cluster import Transformer as Transformer_hypothesis

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        ## Embedding
        if args.frames > 27:
            self.embedding_1 = nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1)
            self.embedding_2 = nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1)
            self.embedding_3 = nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1)
        else:
            self.embedding_1 = nn.Sequential(
                nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1),
                nn.BatchNorm1d(args.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_2 = nn.Sequential(
                nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1),
                nn.BatchNorm1d(args.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_3 = nn.Sequential(
                nn.Conv1d(2*args.out_joints, args.channel, kernel_size=1),
                nn.BatchNorm1d(args.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

        ## SHR & CHI
        self.Transformer_hypothesis = Transformer_hypothesis(3, args.channel, args.d_hid, args.token_num, args.layer_index, length=args.frames)
        
        ## Regression
        self.regression = nn.Sequential(
            nn.BatchNorm1d(args.channel*3, momentum=0.1),
            nn.Conv1d(args.channel*3, 3*args.out_joints, kernel_size=1)
        )

        ## MHG
        self.norm_1 = nn.LayerNorm(args.frames)
        self.norm_2 = nn.LayerNorm(args.frames)
        self.norm_3 = nn.LayerNorm(args.frames)

        self.Transformer_encoder_1 = Transformer_encoder(4, args.frames, args.frames*2, length=2*args.n_joints, h=9)
        self.Transformer_encoder_2 = Transformer_encoder(4, args.frames, args.frames*2, length=2*args.n_joints, h=9)
        self.Transformer_encoder_3 = Transformer_encoder(4, args.frames, args.frames*2, length=2*args.n_joints, h=9)

    def forward(self, x, index=None):
        B, F, J, C = x.shape
        x = rearrange(x, 'b f j c -> b (j c) f').contiguous()

        ## MHG
        x_1 = x   + self.Transformer_encoder_1(self.norm_1(x))
        x_2 = x_1 + self.Transformer_encoder_2(self.norm_2(x_1)) 
        x_3 = x_2 + self.Transformer_encoder_3(self.norm_3(x_2))
        
        ## Embedding
        x_1 = self.embedding_1(x_1).permute(0, 2, 1).contiguous() 
        x_2 = self.embedding_2(x_2).permute(0, 2, 1).contiguous()
        x_3 = self.embedding_3(x_3).permute(0, 2, 1).contiguous()

        ## SHR & CHI
        x, index = self.Transformer_hypothesis(x_1, x_2, x_3, index) 

        ## Regression
        x = x.permute(0, 2, 1).contiguous() 
        x = self.regression(x) 
        x = rearrange(x, 'b (j c) f -> b f j c', j=J).contiguous()

        return x, index

