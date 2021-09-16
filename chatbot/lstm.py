import torch
import torch.nn as nn
import numpy as np
import time
import random
import torch.nn.utils.prune as prune
import warnings
warnings.filterwarnings('ignore')
import copy

class BotModel(nn.Module):
    def __init__(self,input_size, output_size, hidden_size):
        super(BotModel, self).__init__()
        self.relu = nn.ReLU()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size,hidden_size,batch_first=True,dropout=0.2)
        #print(self.lstm._all_weights)
        self.l2 = nn.Linear(hidden_size,hidden_size)
        self.output = nn.Linear(hidden_size,output_size)

    @staticmethod
    def init_hidden(self):
        self.hidden = (torch.zeros(1,self.hidden_size),
                        torch.zeros(1,self.hidden_size))

    def forward(self,x):

        x, self.hidden = self.lstm(x)
        x = x[:,-1,:]
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.output(x)
        return x

#optional pruning in pytorch
def prune_func(model1):
    mod1 = copy.deepcopy(model1).to(device)

    for name,module in mod1.named_modules():
        #print(model.named_modules())
        if isinstance(module,nn.Linear):
            prune.l1_unstructured(module,name='weight',amount = 0.4)

        elif isinstance(module,nn.LSTM):
            prune.l1_unstructured(module,name='weight_hh_l0',amount = 0.2)
            prune.l1_unstructured(module,name='weight_ih_l0',amount = 0.2)
    return mod1

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(device)
    model1 = BotModel(137,40,80)
    model1.to(device)
    model1 = prune_func(model1)
