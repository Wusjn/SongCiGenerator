import json
import pickle
from utils import Lang
import os

lang = Lang()
for i in range(2,9):
    for j in range(2,9):
        filename = "pairs/" + str(i) + "_" + str(j) + ".json"
        with open(filename,"r") as file:
            pairs = json.load(file)
        for pair in pairs:
            lang.addSentence(pair["src"])
            lang.addSentence(pair["trg"])


if not os.path.isdir("data"):
    os.makedirs("data")
with open("data/lang.pkl","wb") as file:
    pickle.dump(lang,file)