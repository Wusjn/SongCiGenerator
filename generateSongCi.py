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
from functools import reduce
import random
import math



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
lang, train_iter, val_iter, test_iter = load_data(16)
vocab_size = len(lang.index2word)

hidden_size = 512
embed_size = 256


encoder = Encoder(vocab_size, embed_size, hidden_size,
                  n_layers=2, dropout=0.5)
decoder = Decoder(embed_size, hidden_size, vocab_size,
                  n_layers=1, dropout=0.5)
seq2seq = Seq2Seq(encoder, decoder)

seq2seq.load_state_dict(torch.load("./.save/seq2seq_{}.pt".format(6)))
seq2seq = seq2seq.cuda()

def printSortedIdx(lang,probs):
    probs = list(map(lambda x:math.exp(x), probs))
    total = sum(probs)
    sortedProbs = sorted(range(len(probs)), key=lambda k: probs[k], reverse=True)
    print(lang.indice2sentence(sortedProbs[:5]))

def getMaxProbIdx(lang,probs):
    printSortedIdx(lang,probs)
    probs = list(map(lambda x:math.exp(x), probs))
    total = sum(probs)
    print(total, max(probs), probs.index(max(probs)))
    choice = random.random() * total
    #print(choice)
    idx, cur = 0,0
    for i,prob in enumerate(probs):
        cur += prob
        if cur >= choice:
            idx = i
            break
    return idx


#TODO: using seq2seq to generate Song Ci

src = "明月几时有"
trg = "把酒问青天不知天上宫阙今夕是何年我欲乘风归去又恐琼楼玉宇高处不胜寒起舞弄清影何似在人间转朱阁低绮户照无眠不应有恨何事长向别时圆人有悲欢离合月有阴晴圆缺此事古难全但愿人长久千里共婵娟"
src = lang.sentence2Indice(src)
trg = lang.sentence2Indice(trg)
src = [1] + src #+ [2]
trg = [1] + trg #+ [2]
src = torch.from_numpy(np.array(src))
trg = torch.from_numpy(np.array(trg))
src = src.unsqueeze(1).cuda()
trg = trg.unsqueeze(1).cuda()
print_tensor(lang,src.cpu())
print_tensor(lang,trg.cpu())

outputs = seq2seq(src, trg, teacher_forcing_ratio=0.0)
outputs = outputs.squeeze(1)

#top1 = outputs.data.max(1)[1]
outputs = outputs.data.detach().cpu().numpy().tolist()
outputs = map(lambda probs: getMaxProbIdx(lang,probs),outputs[1:])

sentence = lang.indice2sentence(outputs)

#print(top1)
print(sentence)
