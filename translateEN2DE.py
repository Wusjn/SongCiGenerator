import os
import math
import argparse
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.nn import functional as F
from model import Encoder, Decoder, Seq2Seq
from utils import *



def translate(model, val_iter, vocab_size, lang):
    model.eval()
    pad = 0
    total_loss = 0
    for b, batch in enumerate(val_iter):
        src = batch["src"]
        trg = batch["trg"]
        src = Variable(src.data.cuda(), volatile=True)
        trg = Variable(trg.data.cuda(), volatile=True)
        output = model(src, trg, teacher_forcing_ratio=0.0)
        loss = F.nll_loss(output[1:].view(-1, vocab_size),
                               trg[1:].contiguous().view(-1),
                               ignore_index=pad)
        total_loss += loss.item()
    return total_loss / len(val_iter)


print("[!] preparing dataset...")
lang, train_iter, val_iter, test_iter = load_data()
vocab_size = len(lang.index2word)

hidden_size = 512
embed_size = 256


encoder = Encoder(vocab_size, embed_size, hidden_size,
                  n_layers=2, dropout=0.5)
decoder = Decoder(embed_size, hidden_size, vocab_size,
                  n_layers=1, dropout=0.5)
seq2seq = Seq2Seq(encoder, decoder).cuda()

seq2seq.load_state_dict(torch.load("./.save/seq2seq_7.pt"))
