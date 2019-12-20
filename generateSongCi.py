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
lang, train_iter = load_data(32)
vocab_size = len(lang.index2word)

hidden_size = 512
embed_size = 256


encoder = Encoder(vocab_size, embed_size, hidden_size,
                  n_layers=2, dropout=0.5)
decoder = Decoder(embed_size, hidden_size, vocab_size,
                  n_layers=1, dropout=0.5)
seq2seq = Seq2Seq(encoder, decoder)

seq2seq.load_state_dict(torch.load("./.save/seq2seq_{}.pt".format(1)))
seq2seq = seq2seq.cuda()

def printSortedIdx(lang,probs):
    probs = list(map(lambda x:math.exp(x), probs))
    total = sum(probs)
    sortedProbs = sorted(range(len(probs)), key=lambda k: probs[k], reverse=True)
    #print(lang.indice2sentence(sortedProbs[:5]))

def getMaxProbIdx(lang,probs):
    printSortedIdx(lang,probs)
    probs = list(map(lambda x:math.exp(x), probs))
    total = sum(probs)
    #print(total, max(probs), probs.index(max(probs)))
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

def sentence2tensor(sentence, lang):
    sentence = [1] + lang.sentence2Indice(sentence)
    sentence = torch.from_numpy(np.array(sentence)).unsqueeze(1)
    #print_tensor(lang,sentence)
    sentence = sentence.cuda()
    return sentence

def outputs2sentence(outputs, lang):
    outputs = outputs.squeeze(1)
    outputs = outputs.data.detach().cpu().numpy().tolist()
    outputs = map(lambda probs: getMaxProbIdx(lang, probs), outputs[1:])
    sentence = lang.indice2sentence(outputs)
    return  sentence

sample = ["明月几时有",
          "把酒问青天",
          "不知天上宫阙",
          "今夕是何年",
          "我欲乘风归去",
          "又恐琼楼玉宇",
          "高处不胜寒",
          "起舞弄清影",
          "何似在人间",
          "转朱阁",
          "低绮户",
          "照无眠",
          "不应有恨",
          "何事长向别时圆",
          "人有悲欢离合",
          "月有阴晴圆缺",
          "此事古难全",
          "但愿人长久",
          "千里共婵娟"]

rhythmic = "水调歌头"
firstSentence = "明月几时有"
sentenceLenth = [5,5,6,5,6,6,5,5,5,3,3,3,4,7,6,6,5,5,5]

def getRhythmicForm(rhythmic):
    regularForm = None
    with open("rhythmics/" + rhythmic + ".sort.json", "r") as file:
        rhythmicForms = json.load(file)
        regularForm = rhythmicForms[0]
    if regularForm == None:
        print("rhythmic not exist!")
        sys.exit(1)
    #print(regularForm)
    return regularForm

def generateSongCi(firstSentence, rhythmic, lang):
    regularForm = getRhythmicForm(rhythmic)
    lines = [firstSentence]
    src = sentence2tensor(firstSentence, lang)
    for trgLenIdx in range(1,len(regularForm["lengths"])):
        trgLen = regularForm["lengths"][trgLenIdx]
        trg = sentence2tensor("人"*trgLen, lang)

        outputs = seq2seq(src, trg, teacher_forcing_ratio=0.0)

        sentence = outputs2sentence(outputs, lang)
        lines.append(sentence)
        src = sentence2tensor(sentence,lang).cuda()
    generatedSongCi = {}
    generatedSongCi["rhythmic"] = regularForm["rhythmic"]
    generatedSongCi["lines"] = lines
    generatedSongCi["punctuations"] = regularForm["punctuations"]

    generatedSongCi["text"] =  generatedSongCi["rhythmic"] + "\n\n"
    for i in range(0,len(lines)):
        generatedSongCi["text"] += lines[i] + generatedSongCi["punctuations"][i] + "\n"
    return generatedSongCi


print(generateSongCi(firstSentence, rhythmic, lang)["text"])