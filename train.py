import pickle
import torch

import torch.nn as nn
import torch.optim as optim
import spacy
from spacy.cli.download import download
import torch.nn.functional as F
from torchtext.data.metrics import bleu_score
from torch.utils.tensorboard import SummaryWriter

import math
import time
from helper import *
import os
import json

import Transformer
from TransformerDataLoader import *



class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

def preparation():
    global device
    global src_vocab
    global trg_vocab
    global train_dataset
    global valid_dataset
    global test_dataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # download(model="en_core_web_sm")
    # download(model="de_core_news_sm")

    save_dir = './wmt17_en_de_processed.pkl'
    with open(save_dir, 'rb') as f:
        data = pickle.load(f)

    train_dataset, valid_dataset, test_dataset = data['train'],data['valid'],data['test']

    src_vocab, trg_vocab = data['vocab']['src'], data['vocab']['trg']

    print(f"training dataset: {len(train_dataset)}")
    print(f"validation dataset: {len(valid_dataset)}")
    print(f"testing dataset: {len(test_dataset)}")

    print(f"src_vocab: {len(src_vocab)}")
    print(f"trg_vocab: {len(trg_vocab)}")

    print(train_dataset[0][0])

def check_batch_data():
    global train_iterator
    global valid_iterator
    global test_iterator

    BATCH_SIZE = 16
    data_path = "./wmt17_en_de_processed.pkl"

    num_workers = 3
    train_iterator = TransformerDataLoader(data_path, 'train', batch_size=BATCH_SIZE,num_workers=num_workers)
    valid_iterator = TransformerDataLoader(data_path, 'valid', batch_size=BATCH_SIZE,num_workers=num_workers)
    test_iterator = TransformerDataLoader(data_path, 'test', batch_size=BATCH_SIZE,num_workers=num_workers)

    # # check for first batch
    # for i, batch in enumerate(train_iterator):
    #     src, trg = next(iter(train_iterator))
    #
    #     print(f"first batch size: {src.shape}")
    #     print(f"first batch size: {trg.shape}")
    #
    #     for i in range(src.shape[1]):
    #         print(f"{i}: {src[0][i].item()}") # the first src sentence
    #
    #     break

def train_model():
    global dim_model
    global dim_hidden
    global dim_vocab
    global num_layers
    global num_heads

    # save the losses
    losses = {}

    dim_model = 512
    dim_hidden = 2048
    dim_vocab = len(trg_vocab)
    num_layers = 6
    num_heads = 8

    model = Transformer.TransformerModel(dim_model, dim_hidden, dim_vocab, N=num_layers, h=num_heads).to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model)} trainable parameters')

    def initialize_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)

    model.apply(initialize_weights)

    # to index(with padding)


    # Adam optimizer
    LEARNING_RATE = 0.0005
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ignore the padding index
    criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

    optimizer = NoamOpt(dim_model, 1, 4000,
                        torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9))

    N_EPOCHS = 10
    CLIP = 1
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        start_time = time.time() # record the start time

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time() # record the end time
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model/transformer.pt')

        test_loss = evaluate(model, test_iterator, criterion)

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(min(train_loss, 100)):.3f}')
        print(f'\tValidation Loss: {valid_loss:.3f}')
        print(f'Test Loss: {test_loss:.3f}')

        save_model(model, epoch)

        losses['train_loss_{}'.format(epoch)] = train_loss
        losses['train_PPL_{}'.format(epoch)] = math.exp(min(train_loss, 100))
        losses['valid_loss_{}'.format(epoch)] = valid_loss
        losses['valid_PPL_{}'.format(epoch)] = math.exp(min(valid_loss, 100))
        losses['valid_bleu_{}'.format(epoch)] = show_bleu(valid_dataset,src_vocab,trg_vocab,model,device)
        losses['test_loss_{}'.format(epoch)] = test_loss
        losses['test_PPL_{}'.format(epoch)] = math.exp(min(test_loss, 100))
        losses['test_bleu_{}'.format(epoch)] = show_bleu(test_dataset,src_vocab,trg_vocab,model,device)

        # dump into json file
        b = json.dumps(losses)
        f2 = open('losses.json', 'w')
        f2.write(b)
        f2.close()

# train function
def train(model, iterator, optimizer, criterion, clip=1):
    model.train() # set as train model
    epoch_loss = 0

    # iterate through whole data set
    for i, batch in enumerate(iterator):
        print("-------------batch{}-------------".format(i))
        src, trg = next(iter(iterator))
        src = src.to(device)
        trg = trg.to(device)

        optimizer.optimizer.zero_grad()

        trg_in = trg[:,:-1].clone().to(device)
        trg_y = trg[:,1:].clone().to(device)
        # exclude the end of the sentence
        # start with the beginning of the sentence
        src_padding_mask = Transformer.get_padding_mask(src, PAD_IDX).to(device)
        trg_padding_mask = Transformer.get_padding_mask(trg_in, PAD_IDX).to(device)
        peek_mask = Transformer.get_peek_mask(trg_in).to(device)
        output = model(src, trg_in ,src_padding_mask,trg_padding_mask,peek_mask)

        # output: [batch size, trg_len - 1, output_dim]
        # trg: [batch size, trg_len]
        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        # Exclude index 0 (<sos>) of output word
        trg_y = trg_y.contiguous().view(-1)

        # output: [batch size * trg_len - 1, output_dim]
        # trg: [batch size * trg len - 1]

        print(output.shape)
        print(trg_y.shape)

        # calculate the loss
        loss = cal_loss(output,trg_y,PAD_IDX,True)
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # update the parameters
        optimizer.step()

        # calculate total loss
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# evaluation
def evaluate(model, iterator, criterion):
    model.eval() # change to evaluation mode
    epoch_loss = 0

    with torch.no_grad():
        # Checking the entire evaluation data
        for i, batch in enumerate(iterator):
            src, trg = next(iter(iterator))
            src = src.to(device)
            trg = trg.to(device)

            trg_in = trg[:,:-1].clone().to(device)
            trg_y = trg[:,1:].clone().to(device)

            # exclude the end of the sentence
            # start with the beginning of the sentence
            src_padding_mask = Transformer.get_padding_mask(src, PAD_IDX).to(device)
            trg_padding_mask = Transformer.get_padding_mask(trg_in, PAD_IDX).to(device)
            peek_mask = Transformer.get_peek_mask(trg_in).to(device)
            output = model(src, trg_in,src_padding_mask,trg_padding_mask,peek_mask)

            # output: [batch size, trg_len - 1, output_dim]
            # trg: [batch size, trg_len]
            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            # Exclude index 0 (<sos>) of output word
            trg_y = trg_y.contiguous().view(-1)

            # output: [batch size * trg_len - 1, output_dim]
            # trg: [batch size * trg len - 1]

            # calculate the loss
            # loss = criterion(output, trg_y)
            loss = cal_loss(output,trg_y,PAD_IDX,True)

            # calculate total loss
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def cal_loss(pred,gold,trg_pad_idx,smoothing=True):
    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def save_model(model, step):
    save_path = os.path.join('model/transformer_{}.pt'.format(step+1))
    torch.save(model.state_dict(), save_path)
    print('Saved model checkpoints into {}...'.format(save_path))

def load_model(resume_iters):
    print('Loading the trained models from step {}...'.format(resume_iters))
    model_path = os.path.join('model/transformer_{}.pt'.format(resume_iters+1))

    ################################################
    model = Transformer.TransformerModel(dim_model, dim_hidden, dim_vocab, N=num_layers, h=num_heads).to(device)
    model.load_state_dict(torch.load(model_path))
    ################################################

    return model


# translation
def translate_sentence(sentence, src_field, trg_field, model, device, max_len=2000, logging=True):
    model.eval() # change into the evaluation mode

    if isinstance(sentence, str):
        nlp = spacy.load("en_core_web_sm")
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # append <sos> token at the beginning and <eos> token at the end
    tokens = [special_symbols[2]] + tokens + [special_symbols[3]]
    if logging:
        print(f"full source token: {tokens}")

    src_indexes = src_field.lookup_indices(tokens)
    if logging:
        print(f"source sentence index: {src_indexes}")

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    # Make sure you have only one <sos> token at first
    trg_indexes = [BOS_IDX]

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        # generate our output
        src_padding_mask = Transformer.get_padding_mask(src, PAD_IDX).to(device)
        trg_padding_mask = Transformer.get_padding_mask(trg_tensor, PAD_IDX).to(device)
        peek_mask = Transformer.get_peek_mask(trg_tensor).to(device)
        output = model(src, trg_tensor,src_padding_mask,trg_padding_mask,peek_mask)

        pred_token = output.argmax(2)[:,-1].item()
        trg_indexes.append(pred_token) # add to output statement

        # The moment you meet <eos>, it ends
        if pred_token == EOS_IDX:
            break

    #  Convert each output word index to an actual word
    trg_tokens = trg_field.lookup_tokens(trg_indexes)

    # Returns the output statement excluding the first <sos>
    return trg_tokens[1:]

def show_bleu(data, src_field, trg_field, model, device, max_len=50):
    trgs = []
    pred_trgs = []
    index = 0

    for d in data:
        src = d[0].to(device)
        trg = d[1].to(device)

        pred_trg = translate_sentence(src, src_field, trg_field, model, device, max_len, logging=False)

        # Remove the last <eos> token
        pred_trg = pred_trg[:-1]

        pred_trgs.append(pred_trg)
        trgs.append([trg])

        # # show the prediction and the target
        # index += 1
        # if (index + 1) % 100 == 0:
        #     print(f"[{index + 1}/{len(data)}]")
        #     print(f"prediction: {pred_trg}")
        #     print(f"target: {trg}")

    bleu = bleu_score(pred_trgs, trgs, max_n=4, weights=[0.25, 0.25, 0.25, 0.25])
    print(f'Total BLEU Score = {bleu*100:.2f}')

    individual_bleu1_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1, 0, 0, 0])
    individual_bleu2_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[0, 1, 0, 0])
    individual_bleu3_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[0, 0, 1, 0])
    individual_bleu4_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[0, 0, 0, 1])

    print(f'Individual BLEU1 score = {individual_bleu1_score*100:.2f}')
    print(f'Individual BLEU2 score = {individual_bleu2_score*100:.2f}')
    print(f'Individual BLEU3 score = {individual_bleu3_score*100:.2f}')
    print(f'Individual BLEU4 score = {individual_bleu4_score*100:.2f}')

    cumulative_bleu1_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1, 0, 0, 0])
    cumulative_bleu2_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1/2, 1/2, 0, 0])
    cumulative_bleu3_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1/3, 1/3, 1/3, 0])
    cumulative_bleu4_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1/4, 1/4, 1/4, 1/4])

    print(f'Cumulative BLEU1 score = {cumulative_bleu1_score*100:.2f}')
    print(f'Cumulative BLEU2 score = {cumulative_bleu2_score*100:.2f}')
    print(f'Cumulative BLEU3 score = {cumulative_bleu3_score*100:.2f}')
    print(f'Cumulative BLEU4 score = {cumulative_bleu4_score*100:.2f}')

    return bleu

if __name__ == "__main__":
    preparation()
    check_batch_data()
    train_model()
    # model = load_model(10)
    # show_bleu(test_dataset,src_vocab,trg_vocab,model,device)

