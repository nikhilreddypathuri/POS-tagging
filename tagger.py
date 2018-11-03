# Team name: Evil Geniuses
# Authors:
# Vijayasaradhi Muthavarapu (Vijay)
# Nikhil Reddy Pathuri (Nikhil)
# Dilip Molugu (Dilip)
# Date: 10/17/2018

#1. This program helps in performing POS tagging for an untagged data. It takes tagged data as input for training the model and an untagged data on which it predicts the 
# POS tags.

#2. Inorder to make this program work first we need to install the required libraries. Then in the command line arguments you need to pass a minimum of 3 arguments. 
#   -> The first argument should be the training data with tagged words.
#   -> The second argument should be the testing data with untagged words.
#   -> The third argument is the standard output file to which our program prints the results to.


# An example to run this program:
# >python tagger.py pos-train.txt pos-test.txt > pos-test-with-tags.txt
# Sample Output:
# No/RB ,/, it/PRP was/VBD n't/RB Black/NNP Monday/NNP ./. But/CC while/IN the/DT....

#3. About our code and algorithm:
# Training data: We used the text file provided to us with pre-tagged data for training.
# Program Logic: 
# Step 1: First our program reads train and test text files and appends the files into a variable. We also remove few unwanted characters like '\n','[',']' from the text.
# Step 2: We tokenize tags from words using regular expressions and convert the data into a 2d list.
# Step 3: Calculate the frequency distribution of the tags.
# Step 4: Calculate the frequency distribution of a tag corresponding to a word using the groupby command.
# Step 5: Calculate the frequency distributoin of tag with its previous tag.
# Step 6: Clean and tokenize the test data for prediction.
# Step 7: Predict the tags by calculating most likely probabilites of words and tags and also tags with previous tags. The Base accuracy = 85.56%
# Step 8: Apply rules on the predictions: 
#    1) #rule1 : if tag is '/NNPS' make it '/NNP' --> Accuracy = 85.63%
#    2) : if word is int make tag to /CD --> Accuracy = 86.39%
#    3) if prev is (/NN or /NNS) current is (/IN with /WDT) next is (/VBD or /VBZ or /VBP) --> Accuracy = 86.47%
#    4) words with $ will have /$ tag --> Accuracy = 86.48%
#    5) words with ( will have /( tag --> Accuracy = 86.49%
#
# Step 9: Combine the predicted tags with words and print to the standard output file.
# Key Feature:We used the formula: P(wi/ti)*P(ti/ti-1) for calculating the likelihood.

import sys
import re
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from random import random
import numpy as np
import pandas as pd
import csv

def pre_processing(arg):
    train = open(sys.argv[arg], "r").read().rstrip().replace('\n', '').replace('[', ''). replace(']', '')
    train1 = re.sub(' +',' ',train)
    return train1

train_text = pre_processing(1)
test_text = pre_processing(2)

#Split by space
train_text = train_text.strip().split(" ")

#identify tag and add space
#split by space for 2d list with word and its corresponding tag
train_text2=[]
for row in train_text:
    row=re.sub(r'(\/[A-Z]*(\|?)[^\d|^/]+)$',r' \1',row)
    train_text2.append(row.strip().split(" "))

#dataframe for easy interpretation
cols = ["words","tags"]
df = pd.DataFrame(data=train_text2, columns=cols)

#Frequency of tags (tn)
fdist_tags = FreqDist(df["tags"])

#Frequency of tags corresponding to the words (wordn,t1...tn)
word_tag=df.groupby(["words","tags"]).size()

#Frequency of tag with its previous tag (tn-1, tn)
two_tags =[]
for x in range(len(df)-1):
    two_tags.append(str(df["tags"][x])+" "+str(df["tags"][x+1]))

fdist_two_tags = FreqDist(two_tags)

#testing
#Tokenize test data
test_tokens = test_text.strip().split(" ")

#Output logic
output=[]
length = len(test_tokens)
first=1
for item in test_tokens:
    if(first==1):
        if(item in word_tag):
            output.append(max(word_tag[item].to_dict()))
            prev_tag=max(word_tag[item].to_dict())
        else:
            output.append("/NN")
            prev_tag="/NN"
        first=0
        continue
    else:
        if(item in word_tag):
            prob_list=[]
            for postag in word_tag[item].to_dict():
                prob=(word_tag[item][postag]/fdist_tags[postag])*(fdist_two_tags[prev_tag+" "+postag]/fdist_tags[prev_tag])
                prob_list.append(prob)
            output.append(list(word_tag[item].to_dict().keys())[prob_list.index(max(prob_list))])
            prev_tag= list(word_tag[item].to_dict().keys())[prob_list.index(max(prob_list))]
        else:
            output.append("/NN")
            prev_tag="/NN"

#combining words with tags
for i in range(len(output)):
    output[i]=output[i]+" "+test_tokens[i]

output2=[]
for row in output:
    output2.append(row.strip().split(" "))

#base accuracy = 85.56%
#rule1 : if tag is '/NNPS' make it '/NNP' --> Accuracy = 85.63%
for i in range(len(output2)):
    if(output2[i][0]=="/NNPS"):
        output2[i][0]="/NNP"
#rule2 : if word is int make tag to /CD --> Accuracy = 86.39%
for i in range(len(output2)):
    for line in output2[i][1]:
        if re.search(r"^(\d)+$",line,re.I):
            output2[i][0]="/CD"
#rule 3 if prev is (/NN or /NNS) current is (/IN with /WDT) next is (/VBD or /VBZ or /VBP) --> Accuracy = 86.47%
for i in range(1,len(output2)-1):
    if((output2[i][0]=="/IN") and (output2[i-1][0]=="/NN" or output2[i-1][0]=="/NNS") and (output2[i+1][0]=="/VBD" or output2[i+1][0]=="/VBZ" or output2[i+1][0]=="/VBP")):
        output2[i][0]="/WDT"
#rule 4 words $ will have /$ tag --> Accuracy = 86.48%
for i in range(len(output2)):
    for line in output2[i][1]:
            if (line=="$"):
                output2[i][0]="/$"
#rule 5 words ( will have /( tag --> Accuracy = 86.49%
for i in range(len(output2)):
    for line in output2[i][1]:
            if (line=="("):
                output2[i][0]="/("

#output
for i in range(len(output2)):
    sys.stdout.write(output2[i][1]+output2[i][0]+" ")

