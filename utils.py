from functools import reduce
import pickle
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import json

class Lang:
    def __init__(self):
        self.word2index = {"<pad>":0, "<SOS>":1, "<EOS>":2}
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
    def partition(self, pairs, batch_size):
        batches = []
        self.batch_size = batch_size
        for begin in range(0,len(pairs),batch_size):
            end = begin + batch_size
            if end > len(pairs):
                end = len(pairs)
            batches.append(pairs[begin:end])
        return batches

    def __init__(self,root,lang):
        self.lang = lang
        self.root = root
        self.batches = []
        for i in range(2,9):
            for j in range(2,9):
                filename = str(i) + "_" + str(j) + ".json"
                with open(filename,"r") as file:
                    pairs = json.load(file)
                self.batches.extend(self.partition(pairs))

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        batch = self.batches[idx]
        src, trg = [], []
        for pair in batch:
            src.append([self.lang.word2index["<SOS>"]] + self.lang.sentence2Indice(pair["src"]))
            trg.append([self.lang.word2index["<SOS>"]] + self.lang.sentence2Indice(pair["trg"]))
        src, trg = np.array(src).transpose(1,0), np.array(trg).transpose(1,0)
        return {"src":src, "trg":trg}

def pad_tensor(vecs):
    max_lenth = max([len(vec) for vec in vecs])
    paded_vecs = []
    for vec in vecs:
        paded_vecs.append(vec + [0 for i in range(max_lenth - len(vec))])

    return np.transpose(np.array(paded_vecs))

def collate_fn(items):
    return {"src":items[0]["src"], "trg":items[0]["trg"]}

def getDataloader(dataset,lang,batch_size):
    return DataLoader(SongCiDataset(dataset,lang,batch_size), batch_size=1, shuffle=True, collate_fn=collate_fn())

def load_data(batch_size):
    with open("data/lang.pkl", "rb") as file:
        lang = pickle.load(file)
    return lang, getDataloader("pairs",lang,batch_size)


def print_tensor(lang,tensor):
    lists = tensor.data.detach().numpy().transpose(1,0).tolist()
    for list in lists:
        print(lang.indice2sentence(list))
    print()