import torch
import torch.nn as nn
import json
import random
from model import BotModel
from chatbot import bag_of_words
import nltk
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#path=r'ausm_intents.json'
path = input("Enter path to dataset: ")
with open(path,'rb') as f:
    data=json.load(f,strict=False)

#file1 = 'nn2_model.pth'
file1 = input("Enter path to model's weights: ")
model = torch.load(file1)

input_size = model['input_size']
hidden_size = model['hidden_size']
output_size = model['output_size']

words = model['words']
labels = model['tags']
mod_st = model['model_state']

modela = BotModel(input_size, output_size, hidden_size).to(device)
modela.load_state_dict(mod_st)
modela.eval()

name = 'Han'
def bot_reply(modela,labels,data):
    print(name + ': Welcome')
    while True:
        sent = input('User: ')
        if sent == 'exit':
            break
        pre = nltk.word_tokenize(sent)

        b = bag_of_words(pre,words)

        pred = b.reshape(1,b.shape[0])
        pred = torch.from_numpy(pred).to(dtype=torch.float32).to(device)
        pred = pred.reshape(1,1,137)

        h = (torch.zeros(1,hidden_size), torch.zeros(1,hidden_size))
        outp = modela(pred,h)

        i,j = torch.max(outp,dim=1)

        tag = labels[j.item()]
        for i in data['intents']:
            if tag == i['tag']:
                resp = i['responses']
                break
        print('Han: '+ random.choice(resp))


bot_reply(modela,labels,data)
