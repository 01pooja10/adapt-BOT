import json
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import torch
from torch.cuda.amp import autocast, GradScaler
from data import preprocess_data,BotData
from lstm import BotModel, prune_func
from hyperparams import hyperparams
import time

#path=r'C:\Users\Pooja\Documents\.py_files\ausm_intents.json'
path = input("Enter path to dataset: ")

with open(path,'rb') as f:
    data=json.load(f,strict=False)

xtrain, ytrain, words, labels = preprocess_data(data)
input_size, output_size, num_layers, hidden_size, learning_rate = hyperparams(xtrain,ytrain)

#set grad scaler instance
scaler = GradScaler()

#collect dataset
df=BotData()
train_loader=DataLoader(dataset=df,batch_size=8,shuffle=True)

#load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mod = BotModel(input_size, output_size, hidden_size).to(device)
model = prune_func(mod)

#set loss criterion and optimizer configurations
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 100

s = time.time()
for epoch in range(epochs):
    for (w,l) in train_loader:
        w = w.reshape(8,1,137)
        w = w.to(dtype=torch.float32).to('cuda:0')
        l = l.to(dtype=torch.long).to('cuda:0')

        optimizer.zero_grad()
        with autocast():
            outputs = model(w).to(device)
            loss = criterion(outputs,l)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if epoch % 10 == 0:
        print('epoch: '+str(epoch)+', loss: '+str(loss.item()))

e = time.time()
print('Final loss: '+str(loss.item()))
print('Training done')
print('Time to train: ',(e-s))

#save the model details
model_data = {
            'model_state': model.state_dict(),
            'input_size': input_size,
            'output_size': output_size,
            'hidden_size': hidden_size,
            'words': words,
            'tags': labels
}

to_save = 'your_model.pth'
torch.save(model_data, to_save)
