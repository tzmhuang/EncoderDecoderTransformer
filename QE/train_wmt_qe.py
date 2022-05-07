import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from scipy import stats

from QEDataLoader import *

torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


import Transformer

class Mlp(nn.Module):
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

class TransformerQE(nn.Module):
    def __init__(self, dim_model, dim_hidden, src_dim_vocab, trg_dim_vocab, mlp_hidden, N=6, h=8, dropout=0.1):
        super().__init__()
        self.encoder = Transformer.Encoder(N, dim_model, dim_hidden, h, dropout)
        self.decoder = Transformer.Decoder(N, dim_model, dim_hidden, h, dropout)
        self.src_embedding = Transformer.TokenEmbedding(src_dim_vocab, dim_model)
        self.trg_embedding = Transformer.TokenEmbedding(trg_dim_vocab, dim_model)
        self.enc_positional_encoding = Transformer.PositionalEncoding(dim_model, dropout=dropout)
        self.dec_positional_encoding = Transformer.PositionalEncoding(dim_model, dropout=dropout)
        self.linear1 = nn.Linear(mlp_hidden, mlp_hidden)
        self.output = Mlp(dim_model, mlp_hidden*2, 1, drop=0.1)
        
    def encode(self, X, padding_mask):
        src_emb = self.enc_positional_encoding(self.src_embedding(X))
        return self.encoder(src_emb, padding_mask)

    def decode(self, X, E, src_padding_mask, trg_padding_mask, peek_mask):
        trg_emb = self.dec_positional_encoding(self.trg_embedding(X))
        return self.decoder(trg_emb, E, src_padding_mask, trg_padding_mask, peek_mask)
        
    def forward(self, src, trg, src_padding_mask, trg_padding_mask, peek_mask):
        enc = self.encode(src, src_padding_mask)
        dec = self.decode(trg, enc, src_padding_mask, trg_padding_mask, peek_mask).mean(1)
        out = self.linear1(dec)#.mean(dim=1, keepdim=False)
        return self.output(out) # output are logits


def process_data(data, tok, voc):
    emb_tok = tok(data.lower())
    emb = [2] + [voc.vocab.stoi[t.lower()] for t in emb_tok] + [3]

    return emb

def train(iterator, model, loss_fn, optimizer, epoc):
    model.train()
    for i in range(epoc):
        epoch_loss = 0.0
        count = 0
        for batch, (src, trg, score) in enumerate(iterator):
            src = src.to(device)
            trg = trg.to(device)
            src_padding_mask = Transformer.get_padding_mask(
                src , 1).to(device)
            trg_padding_mask = Transformer.get_padding_mask(
                trg, 1).to(device)
            B, L = trg.size()
            # peek_mask = Transformer.get_peek_mask(trg).to(device)
            peek_mask = (torch.zeros(B, L, L)>0).to(device)
            output = model(src,trg,src_padding_mask,trg_padding_mask,peek_mask).view(-1)
            score = score.to(device)
            # print(score.shape)
            loss = loss_fn(output, score.view(-1))
            epoch_loss = epoch_loss + loss.item()
            count = count + 1
            if count % 5 == 0:
                print(f"Epoch: {i}, Loss in the {batch}/{len(iterator)} batch is " + str(loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Loss in the {batch}/{len(iterator)} batch is " + str(epoch_loss / count))            


def test(iterator, model, loss_fn):
    model.eval()
    epoch_loss = 0
    count = 0
    res = torch.ones(1).to(device)
    
    with torch.no_grad():
        for batch, (src, trg, score) in enumerate(iterator):
            src = src.to(device)
            trg = trg.to(device)
            src_padding_mask = Transformer.get_padding_mask(
                src , 1).to(device)
            trg_padding_mask = Transformer.get_padding_mask(
                trg, 1).to(device)
            B, L = trg.size()
            # peek_mask = Transformer.get_peek_mask(trg).to(device)
            peek_mask =  (torch.zeros(B, L, L)>0).to(device)
            output = model(src,trg,src_padding_mask,trg_padding_mask,peek_mask).view(-1)
            res = torch.cat((res,output)) 
            # print(output)
            # print(type(output))
            # print(type(score_batch))
            score = score.to(device)
            loss = loss_fn(output, score.view(-1))
            epoch_loss = epoch_loss + loss.item()
            count = count + 1
    print("Loss in the " + str(count) + " batch is " + str(epoch_loss / count))
    return res

def main():
    # Create data loaders.
    print("Loading iterator")
    data_path = "./mlqe_en_de_processed_bpe.pkl"

    batch_size = 128

    train_iterator =  QEDataLoader(data_path, 'train', batch_size=batch_size, num_workers=4, use_bpe=True)
    valid_iterator =  QEDataLoader(data_path, 'valid', batch_size=batch_size, num_workers=4, use_bpe=True)
    test_iterator =  QEDataLoader(data_path, 'test', batch_size=batch_size, num_workers=4, use_bpe=True)


    src_vocab_dim = len(train_iterator.dataset.vocab['src'])
    trg_vocab_dim = len(train_iterator.dataset.vocab['trg'])
    dim_model = 128
    dim_hidden = 2048
    num_layers = 6
    num_heads = 8

    loss_fn = nn.MSELoss()
    model = TransformerQE(dim_model, dim_hidden, src_vocab_dim, trg_vocab_dim, N=num_layers, h=num_heads, dropout=0.1, 
                    mlp_hidden = dim_model).to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # pretrained_dict = torch.load('transformer_best.ckpt', map_location=torch.device('cpu'))
    pretrained_dict = torch.load('./QE/transformer_best.ckpt')

    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    print("loading model")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print("training")

    train(train_iterator, model, loss_fn, optimizer, epoc=1)

    print("testing")
    res = test(test_iterator, model,loss_fn)
    res = res[1:]
    res = res.cpu()
    res = res.detach().numpy()

    print("Evaluating result")
    test_zscore = [p[2] for p in test_iterator.dataset]
    pr = stats.pearsonr(res,test_zscore)
    sp = stats.spearmanr(res,test_zscore)
    print("PR: ", pr)
    print("SP: ", sp)

if __name__ == "__main__":
    main()