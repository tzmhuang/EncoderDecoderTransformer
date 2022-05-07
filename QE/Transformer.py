import os
import math
import datetime
import time
import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
from torch.backends import cudnn

from helper import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_peek_mask(X):
    B, L = X.size()
    mask = (torch.ones(1, L,L)).triu(1)
    return mask > 0

def get_padding_mask(X, pad_token):
    mask = (X == pad_token) 
    return mask.unsqueeze(1)

class ScaledDPAttention(nn.Module):
    def __init__(self, dim_scale):
        super().__init__()
        self.scale = math.sqrt(dim_scale)
        self.sm = nn.Softmax(dim=-1)
    
    def forward(self, q, k, v, mask):
        output = torch.matmul(q, k.transpose(-2, -1))
        output = torch.div(output, self.scale)
        # mask = (-1*torch.ones(L,L)*float('inf')).triu(1)
        output = output.masked_fill(mask, -1*float('inf')) # masked if True; attention + padding
#        output = output + q.size()[1]
        output = self.sm(output)
        output = torch.matmul(output, v)
        return output


class RPScaledDPAttention(nn.Module):
    def __init__(self, dim_head, max_len = 150):
        super().__init__()
        self.scale = math.sqrt(dim_head)
        self.max_len = max_len
        self.pos_embedding = nn.Parameter(torch.zeros(2*max_len + 1, dim_head))
        self.sm = nn.Softmax(dim=-1)
    
    def forward(self, q, k, v, mask):
        B, H, Lq, Dh = q.size()
        Lk = k.size(2)
        R = self.relative_pos(Lq,Lk) # lq, lk, dh
        q_t = q.permute(2, 0, 1, 3).contiguous().view(Lq, B*H, Dh) # lq, b*h, dh
        Rp = torch.matmul(q_t, R.transpose(1,2)).transpose(0,1)
        Rp = Rp.contiguous().view(B, H, Lq, Lk)
        output = torch.matmul(q, k.transpose(-2, -1)) + Rp

        output = torch.div(output, self.scale)
        # mask = (-1*torch.ones(L,L)*float('inf')).triu(1)
        output = output.masked_fill(mask, -1*float('inf')) # masked if True; attention + padding
#        output = output + q.size()[1]
        output = self.sm(output)
        output = torch.matmul(output, v)
        return output
    
    def relative_pos(self, Lq, Lk):
        R = torch.arange(Lk).unsqueeze(0) - torch.arange(Lq).unsqueeze(1) # Lq x Lk
        R_cropped = torch.clamp(R, -self.max_len, self.max_len)
        R_cropped = R_cropped + self.max_len-1 
        R_cropped = R_cropped.to(device)
        return self.pos_embedding[R_cropped] # Lq x Lk x Dh


    

# class MultiHeadAttention(nn.Module):
#     def __init__(self, dim_model, h, dropout=0.1): # dim_key, dim_val = dim_model/h (?)
#         super().__init__()
#         self.h = h
#         self.dim_model = dim_model
#         self.dim_head = dim_model // h
    
#         linear_layer = nn.Linear(in_features=dim_model, out_features=self.dim_head, bias=False)
#         attention_layer = ScaledDPAttention(self.dim_head)

#         self.v_layers = nn.ModuleList([copy.deepcopy(linear_layer) for _ in range(self.h)])
#         self.k_layers = nn.ModuleList([copy.deepcopy(linear_layer) for _ in range(self.h)])
#         self.q_layers = nn.ModuleList([copy.deepcopy(linear_layer) for _ in range(self.h)])
#         self.attention_layers = nn.ModuleList([copy.deepcopy(attention_layer) for _ in range(self.h)])

#         self.linear = nn.Linear(in_features = dim_model, out_features=dim_model, bias=False)
#         self.dropout = nn.Dropout(p=dropout)
    
#     def forward(self, Q, K, V, mask):
#         outs = []
#         for i in range(self.h):
#             q = self.q_layers[i](Q)
#             k = self.k_layers[i](K)
#             v = self.v_layers[i](V)
#             o = self.attention_layers[i](q, k, v, mask)
#             outs.append(o)
#         output = torch.cat(outs, dim=-1) # check dimenson
#         return self.dropout(self.linear(output))



class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model, h, dropout=0.1, pos_encoding='static'): # dim_key, dim_val = dim_model/h (?)
        super().__init__()
        self.h = h
        self.dim_model = dim_model
        self.dim_head = dim_model // h
        assert (self.dim_head * h == dim_model)

        self.v_layer = nn.Linear(in_features=dim_model, out_features=self.dim_model, bias=True)
        self.k_layer = nn.Linear(in_features=dim_model, out_features=self.dim_model, bias=True)
        self.q_layer = nn.Linear(in_features=dim_model, out_features=self.dim_model, bias=True)
        if pos_encoding == 'relative':
            self.attention_layer = RPScaledDPAttention(self.dim_head)
        else:
            self.attention_layer = ScaledDPAttention(self.dim_head)

        self.linear = nn.Linear(in_features = dim_model, out_features=dim_model, bias=True)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, Q, K, V, mask):
        b, lq, lk, lv = Q.size(0), Q.size(1), K.size(1), V.size(1)
        q = self.q_layer(Q).view(b, lq, self.h, self.dim_head)
        k = self.k_layer(K).view(b, lk, self.h, self.dim_head)
        v = self.v_layer(V).view(b, lv, self.h, self.dim_head)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        mask = mask.unsqueeze(1)
        attn = self.attention_layer(q, k, v, mask)  # check broadcasting
        attn = attn.transpose(1,2).contiguous().view(b, lq, -1)
        out = self.dropout(self.linear(attn))
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        get_pos = lambda pos : [pos/(10000**(2*(i//2)/dim_model)) for i in range(dim_model)]
        code = np.array([get_pos(i) for i in range(max_len)])
        encoding = np.zeros((max_len, dim_model))
        encoding[:, 0::2] = np.sin(code[:, 0::2])
        encoding[:, 1::2] = np.cos(code[:, 1::2])
        encoding = torch.Tensor(encoding).unsqueeze(0) # for batch broadcast
        self.register_buffer('encoding', encoding)
    
    def forward(self, X):
        X = X + self.encoding[:,:X.size(1)]
        return self.dropout(X)

class PositionalEncodingLearned(nn.Module):
    def __init__(self, dim_model, dropout=0.1, max_len=150):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Embedding(max_len, dim_model)
    
    def forward(self, X):
        B, L, D = X.size()
        position = torch.arange(0, L).unsqueeze(0).repeat(B, 1).to(device)
        X = X + self.pos_embedding(position)
        return self.dropout(X)

class PositionalEncodingPlaceHolder(nn.Module):
    def __init__(self, dim_model, dropout=0.1, max_len=150):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, X):
        return self.dropout(X)

class TokenEmbedding(nn.Module):
    def __init__(self, dim_vocab, dim_emb):
        super().__init__()
        self.embedding = nn.Embedding(dim_vocab, dim_emb, padding_idx = PAD_IDX)
        self.dim_emb = dim_emb
    
    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.dim_emb) # see Paper: 3.4


class EncoderLayer(nn.Module):
    def __init__(self, dim_model, dim_hidden, h=8, dropout=0.1, pos_encoding='static'):
        super().__init__()        
        self.attention = MultiHeadAttention(dim_model, h, dropout, pos_encoding)
        self.FFN = nn.Sequential(
            nn.Linear(in_features=dim_model, out_features=dim_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=dim_hidden, out_features=dim_model)
        )
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, X, padding_mask):
        A = self.attention(Q = X, K = X, V = X, mask=padding_mask)
        A = self.norm1(A + X)
        F = self.FFN(A)
        F = self.dropout(F)
        return self.norm2(F + A)
    

class Encoder(nn.Module):
    def __init__(self, N, dim_model, dim_hidden, h=8, dropout=0.1, pos_encoding='static'):
        super().__init__()
        encoder_layer = EncoderLayer(dim_model, dim_hidden, h, dropout, pos_encoding)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(N)])
        self.norm = nn.LayerNorm(dim_model)

    def forward(self, X, padding_mask):
        for layer in self.layers:
            X = layer(X, padding_mask)
        return self.norm(X)


class DecoderLayer(nn.Module):
    def __init__(self, dim_model, dim_hidden, h=8, dropout=0.1, pos_encoding='static'):
        super().__init__()
        self.masked_attention = MultiHeadAttention(dim_model, h, dropout, pos_encoding)
        # self.masked_attention = nn.MultiheadAttention(dim_model, h, dropout, batch_first=True)
        self.attention = MultiHeadAttention(dim_model, h, dropout, pos_encoding)
        # self.attention = nn.MultiHeadAttention(dim_model, h, dropout, batch_first=True)
        self.FFN = nn.Sequential(
            nn.Linear(in_features=dim_model, out_features=dim_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=dim_hidden, out_features=dim_model)
        )
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, X, encoder_feature, src_mask, trg_mask): # (X, encoder_feature)
        masked_A = self.masked_attention(Q = X, K = X, V = X, mask = trg_mask)
        # masked_A,_ = self.masked_attention(X, X, X, attn_mask = trg_mask)
        masked_A = self.norm1(masked_A + X)
        A = self.attention(Q = masked_A, K = encoder_feature, V = encoder_feature, mask = src_mask)
        # A = self.attention(masked_A, encoder_feature, encoder_feature, key_padding_mask = src_mask)
        A = self.norm2(A + masked_A)
        F = self.FFN(A)
        F = self.dropout(F)
        return self.norm3(F + A)

class Decoder(nn.Module):
    def __init__(self, N, dim_model, dim_hidden, h=8, dropout=0.1, pos_encoding='static'):
        super().__init__()
        decoder_layer = DecoderLayer(dim_model, dim_hidden, h, dropout, pos_encoding)
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(N)])
        self.norm = nn.LayerNorm(dim_model)
    
    def forward(self, X, E, src_padding_mask, trg_padding_mask, peek_mask):
        trg_mask = trg_padding_mask | peek_mask
        src_mask = src_padding_mask
        for layer in self.layers:
            X = layer(X, E, src_mask, trg_mask)
        return self.norm(X)

class Generator(nn.Module):
    def __init__(self, dim_model, dim_output):
        super().__init__()
        self.linear = nn.Linear(in_features=dim_model, out_features = dim_output, bias=True)
        # self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, X, logit=True):
        return self.linear(X)

class TransformerModel(nn.Module):
    def __init__(self, dim_model, dim_hidden, src_dim_vocab, trg_dim_vocab, N=6, h=8, dropout=0.1, weight_sharing=False, pos_encoding='static'):
        super().__init__()

        self.pos_encoding = pos_encoding
        assert self.pos_encoding in ['static', 'learned', 'relative']
        self.encoder = Encoder(N, dim_model, dim_hidden, h, dropout, self.pos_encoding)
        self.decoder = Decoder(N, dim_model, dim_hidden, h, dropout, self.pos_encoding)
        self.src_embedding = TokenEmbedding(src_dim_vocab, dim_model)
        self.trg_embedding = TokenEmbedding(trg_dim_vocab, dim_model)
        self.generator = Generator(dim_model, trg_dim_vocab)

        if self.pos_encoding == 'learned':
            self.enc_positional_encoding = PositionalEncodingLearned(dim_model, dropout=dropout)
            self.dec_positional_encoding = PositionalEncodingLearned(dim_model, dropout=dropout)
        elif self.pos_encoding == 'relative':
            self.enc_positional_encoding = PositionalEncodingPlaceHolder(dim_model, dropout=dropout)
            self.dec_positional_encoding = PositionalEncodingPlaceHolder(dim_model, dropout=dropout)
        else:
            self.enc_positional_encoding = PositionalEncoding(dim_model, dropout=dropout)
            self.dec_positional_encoding = PositionalEncoding(dim_model, dropout=dropout)

        if weight_sharing:
            self.generator.linear.weight = self.trg_embedding.embedding.weight

    
    def encode(self, X, padding_mask):
        src_emb = self.enc_positional_encoding(self.src_embedding(X))
        return self.encoder(src_emb, padding_mask)

    def decode(self, X, E, src_padding_mask, trg_padding_mask, peek_mask):
        trg_emb = self.dec_positional_encoding(self.trg_embedding(X))
        return self.decoder(trg_emb, E, src_padding_mask, trg_padding_mask, peek_mask)
        
    def forward(self, src, trg, src_padding_mask, trg_padding_mask, peek_mask):
        enc = self.encode(src, src_padding_mask)
        dec = self.decode(trg, enc, src_padding_mask, trg_padding_mask, peek_mask)
        return self.generator(dec) # output are logits