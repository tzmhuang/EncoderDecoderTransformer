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
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils import data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.backends import cudnn

# TODO: Add trg_mask, src_padding mask, trg_padding_mask as forward arg


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
        self.scale = math.sqrt(self.dim_scale)
        self.sm = nn.Softmax(dim=-1)
    
    def forward(self, q, k, v, mask):
        B, L, D = q.size()
        output = torch.matmul(q, k.transpose(-2, -1))
        output = torch.div(output, self.scale)
        # mask = (-1*torch.ones(L,L)*float('inf')).triu(1)
        output = output.masked_fill(mask, -1*float('inf')) # masked if True; attention + padding
#        output = output + q.size()[1]
        output = self.sm(output)
        output = torch.matmul(output, v)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model, h): # dim_key, dim_val = dim_model/h (?)
        super().__init__()
        self.h = h
        self.dim_model = dim_model
        self.dim_head = dim_model // h
        
        self.v_layers = []
        self.k_layers = []
        self.q_layers = []
        self.attention_layers = []
        for i in range(self.h):
            self.q_layers.append(nn.Linear(in_features=dim_model, out_features=self.dim_head, bias=False))
            self.k_layers.append(nn.Linear(in_features=dim_model, out_features=self.dim_head, bias=False))
            self.v_layers.append(nn.Linear(in_features=dim_model, out_features=self.dim_head, bias=False))
            self.attention_layers.append(ScaledDPAttention(self.dim_head))
        self.linear = nn.Linear(in_features = h*self.dim_head, out_features=dim_model, bias=False)
    
    def forward(self, Q, K, V, mask):
        outs = []
        for i in range(self.h):
            q = self.q_layers[i](Q)
            k = self.k_layers[i](K)
            v = self.v_layers[i](V)
            o = self.attention_layers[i](q, k, v, mask)
            outs.append(o)
        output = torch.cat(outs, dim=-1) # check dimenson 
        return self.linear(output)


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len=1000):
        super().__init__()
        get_pos = lambda pos : [pos/(10000**(2*(i//2)/dim_model)) for i in range(dim_model)]
        code = np.array([get_pos(i) for i in range(max_len)])
        encoding = np.zeros((max_len, dim_model))
        encoding[:, 0::2] = np.sin(code[:, 0::2])
        encoding[:, 1::2] = np.cos(code[:, 1::2])
        encoding = torch.Tensor(encoding).unsqueeze(0) # for batch broadcast
        self.register_buffer('encoding', encoding)
    
    def forward(self, X):
        X = X + self.encoding[:,:X.size(1)]
        return X

class TokenEmbedding(nn.Module):
    def __init__(self, dim_vocab, dim_emb):
        super().__init__()
        self.embedding = nn.Embedding(dim_vocab, dim_emb)
        self.dim_emb = dim_emb
    
    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.dim_emb) # see Paper: 3.4


class EncoderLayer(nn.Module):
    def __init__(self, dim_model, dim_hidden, h=8):
        super().__init__()        
        self.attention = MultiHeadAttention(dim_model, h)
        self.FFN = nn.Sequential(
            nn.Linear(in_features=dim_model, out_features=dim_hidden),
            nn.ReLU(),
            nn.Linear(in_features=dim_hidden, out_features=dim_model)
        )
        self.norm = nn.LayerNorm(dim_model)
    
    def forward(self, X, padding_mask):
        A = self.attention(Q = X, K = X, V = X, mask=padding_mask)
        A = self.norm(A + X)
        F = self.FFN(A)
        return self.norm(F + A)
    

class Encoder(nn.Module):
    def __init__(self, N, dim_model, dim_hidden, h=8):
        super().__init__()
        encoder_layer = EncoderLayer(dim_model, dim_hidden, h)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(N)])

    def forward(self, X, padding_mask):
        for layer in self.layers:
            X = layer(X, padding_mask)
        return X


class DecoderLayer(nn.Module):
    def __init__(self, dim_model, dim_hidden, h=8):
        super().__init__()
        self.masked_attention = MultiHeadAttention(dim_model, h)
        self.attention = MultiHeadAttention(dim_model, h)
        self.FFN = nn.Sequential(
            nn.Linear(in_features=dim_model, out_features=dim_hidden),
            nn.ReLU(),
            nn.Linear(in_features=dim_hidden, out_features=dim_model)
        )
        self.norm = nn.LayerNorm(dim_model)
    
    def forward(self, X, encoder_feature, src_mask, trg_mask): # (X, encoder_feature)
        masked_A = self.masked_attention(Q = X, K = X, V = X, mask = trg_mask)
        masked_A = self.norm(masked_A + X)
        A = self.attention(Q = masked_A, K = encoder_feature, V = encoder_feature, mask = src_mask)
        A = self.norm(A + masked_A)
        F = self.FFN(A)
        return self.norm(F + A)

class Decoder(nn.Module):
    def __init__(self, N, dim_model, dim_hidden, h=8):
        super().__init__()
        decoder_layer = DecoderLayer(dim_model, dim_hidden)
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(N)])
    
    def forward(self, X, E, src_padding_mask, trg_padding_mask, peek_mask):
        trg_mask = trg_padding_mask & peek_mask
        src_mask = src_padding_mask
        for layer in self.layers:
            X = layer(X, E, src_mask, trg_mask)
        return X

class Generator(nn.Module):
    def __init__(self, dim_model, dim_output):
        super().__init__()
        self.linear = nn.Linear(in_features=dim_model, out_features = dim_output)
        # self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, X, logit=True):
        return self.linear(X)

class Transformer(nn.Module):
    def __init__(self, dim_model, dim_hidden, dim_vocab, N=6, h=8):
        super().__init__()

        self.encoder = Encoder(N, dim_model, dim_hidden, h)
        self.decoder = Decoder(N, dim_model, dim_hidden, h)
        self.src_embedding = TokenEmbedding(dim_vocab, dim_model)
        self.trg_embedding = TokenEmbedding(dim_vocab, dim_model)
        self.generator = Generator(dim_model, dim_vocab)
        self.positional_encoding = PositionalEncoding(dim_model)
    
    def encode(self, X, padding_mask):
        src_emb = self.positional_encoding(self.src_embedding(X))
        return self.encoder(src_emb, padding_mask)

    def decode(self, X, E, src_padding_mask, trg_padding_mask, peek_mask):
        trg_emb = self.positional_encoding(self.trg_embedding(X))
        return self.decoder(trg_emb, E, src_padding_mask, trg_padding_mask, peek_mask)
        
    def forward(self, src, trg, src_padding_mask, trg_padding_mask, peek_mask):
        enc = self.encode(src, src_padding_mask)
        dec = self.decode(trg, enc, src_padding_mask, trg_padding_mask, peek_mask)
        return self.generator(dec) # output are logits