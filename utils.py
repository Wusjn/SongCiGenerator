from functools import reduce
import pickle
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:"<pad>", 1: "<SOS>", 2: "<EOS>"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def sentence2Indice(self, sentence):
        indice = []
        for word in sentence:
            indice.append(self.word2index[word])
        return indice

    def indice2sentence(self, indice):
        sentence = ""
        for index in indice:
            sentence += self.index2word[index]
        return sentence


class SongCiDataset(Dataset):
    def __init__(self,SongCiDatabaseFile):
        self.SongCiDatasets = []
        with open(SongCiDatabaseFile,"rb") as file:
           self.SongCiDatasets = pickle.load(file)


    def __len__(self):
        return len(self.SongCiDatasets)

    def __getitem__(self, idx):
        SongCi = self.SongCiDatasets[idx]
        rhythmic = SongCi["rhythmic"]
        src = SongCi["lines"][0]
        trg = reduce(lambda a, b: a+b, SongCi["lines"][1:], [])
        return {"src":src, "trg":trg, "rhythmic":rhythmic}

def pad_tensor(vecs):
    max_lenth = max([len(vec) for vec in vecs])
    paded_vecs = []
    for vec in vecs:
        paded_vecs.append(vec + [0 for i in range(max_lenth - len(vec))])

    return np.transpose(np.array(paded_vecs))

def collate_fn(items):
    srcs = [item["src"] for item in items]
    trgs = [item["trg"] for item in items]
    rhythmics = [item["rhythmic"] for item in items]

    return {"src":pad_tensor(srcs), "trg":pad_tensor(trgs), "rhythmic":pad_tensor(rhythmics)}

def getDataloader(dataset,batch_size):
    return DataLoader(SongCiDataset(dataset), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

def load_data(batch_size):
    with open("data/lang.pkl", "rb") as file:
        lang = pickle.load(file)
    return lang, getDataloader("data/train_set.pkl",batch_size), getDataloader("data/val_set.pkl",batch_size), getDataloader("data/test_set.pkl",batch_size)