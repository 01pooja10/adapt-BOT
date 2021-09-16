import torch
import torch.nn as nn
import json
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from torch.utils.data import DataLoader,Dataset
import warnings
warnings.filterwarnings('ignore')

#path=r'C:\Users\Pooja\Documents\.py_files\ausm_intents.json'

path = input('Enter root path to the dataset: ')

with open(path,'rb') as f:
    data=json.load(f,strict=False)

lemm=WordNetLemmatizer()

def bag_of_words(sent,words):
    bow = np.zeros(len(words), dtype=np.float32)
    wrd=[lemm.lemmatize(w.lower()) for w in sent]
    for som in wrd:
        for idx,w in enumerate(words):
            if w == som:
                bow[idx]=1
    return np.array(bow)

def preprocess_data(data):
    xtrain,ytrain = [],[]
    words,labels,x,y=[],[],[],[]

    for intent in data['intents']:
        for pattern in intent['patterns']:
            p=nltk.word_tokenize(pattern)
            words.extend(p)
            x.append((p,intent['tag']))
        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    exclude=['!','.','?',',','%','#','&',':',';','/']
    words = [lemm.lemmatize(w.lower()) for w in words if w not in exclude]

    words=sorted(list(words))
    labels=sorted(list(labels))

    train=[]
    output=[0]*len(labels)

    for (p1,p2) in x:

        bow=[]
        wrd=[lemm.lemmatize(w.lower()) for w in p1]
        for w in words:
            if w in wrd:
                bow.append(1)
            else:
                bow.append(0)

        #proc = bag_of_words(p1,words)
        xtrain.append(bow)
        ytrain.append(labels.index(p2))

    xtrain=np.array(xtrain)
    ytrain=np.array(ytrain)
    return xtrain, ytrain, words, labels


xtrain, ytrain,words,labels = preprocess_data(data)

class BotData(Dataset):
    def __init__(self):
        self.l=len(xtrain)
        self.x1=xtrain
        self.y1=ytrain

    def __getitem__(self,index):
        return self.x1[index],self.y1[index]

    def __len__(self):
        return self.l
