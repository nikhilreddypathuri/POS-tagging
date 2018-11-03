import re
import nltk
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from random import random
import numpy as np
import pandas as pd


def pre_processing():
    train = open("pos-train.txt", "r").read().rstrip().replace('\n', '').replace('[', ''). replace(']', '')
    train1 = re.sub(' +',' ',train)
    return train1

sample_text = pre_processing()

# getting all the POS tags from the original text
x = re.findall(r"(?!^\\)\/([A-Z]+\$?)+ ", sample_text)
# freq dist of all the POS tag
tag_freq_dist = FreqDist(x)

# removing the "\n" from the text and replacing it with blank and replacing the square brackets
train2 = sample_text.rstrip().replace('\n', '').replace('[', ''). replace(']', '')
# train2


# replacing all the / with the blanks
textWithoutExtra = re.sub(r'((?!\\)\/)', r' ',train2)
# removing extra spaces
textWithoutExtra = re.sub(' +', ' ',train2)


allTokenize = textWithoutExtra.split()

for w in range(len(allTokenize)):
    if w == len(allTokenize):
        break
    if allTokenize[w] == "PRP" and allTokenize[w+1] == "$":
        allTokenize[w] = "PRP$"
        del allTokenize[w+1]




