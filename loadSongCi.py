import json
import re
import random
import pickle
import numpy
from utils import Lang
import os

if not os.path.isdir("data"):
    os.makedirs("data")

rawSongCiDatabase = []
for i in range(0,22):
    with open("ci/ci.song.{}.json".format(i*1000),"r",encoding="utf-8") as file:
        rawSongCiDatabase.extend(json.load(file))

SongCiDatasets = []
lang = Lang()
for rawSongCi in rawSongCiDatabase:
    SongCi = {}
    lang.addSentence(rawSongCi["rhythmic"])
    SongCi["rhythmic"] = lang.sentence2Indice(rawSongCi["rhythmic"])
    SongCi["lines"] = []
    for paragraph in rawSongCi["paragraphs"]:
        for line in re.split("，|。|、", paragraph):
            if len(line) > 0:
                lang.addSentence(line)
                SongCi["lines"].append(lang.sentence2Indice(line))
    SongCiDatasets.append(SongCi)

with open("data/SongCiDatasets.pkl","wb") as file:
    pickle.dump(SongCiDatasets,file)
with open("data/lang.pkl","wb") as file:
    pickle.dump(lang,file)

random.shuffle(SongCiDatasets)

train_size = int(0.8 * len(SongCiDatasets))
val_size = int(0.1 * len(SongCiDatasets))
test_size = len(SongCiDatasets) - train_size - val_size

train_set = SongCiDatasets[:train_size]
val_set = SongCiDatasets[train_size:train_size+val_size]
test_set = SongCiDatasets[train_size+val_size:]

with open("data/train_set.pkl","wb") as file:
    pickle.dump(train_set,file)
with open("data/val_set.pkl","wb") as file:
    pickle.dump(val_set,file)
with open("data/test_set.pkl","wb") as file:
    pickle.dump(test_set,file)