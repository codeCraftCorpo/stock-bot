import model
from config import getConfigs
from model import Encoder,Decoder,InputEmbeddings,PositionalEncoding,ProjectionLayer,FeedForwardBlock, build_stock_transformer
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import os
from pprint import pprint
import numpy as np
from akshares_getdata import getDataLoader

ak_config, transformer_config = getConfigs()

#trainModel
def trainModel (trainloader :DataLoader ,model: model, model_config: dict):
    
    model_file_path = os.path.join(model_config["model_folder"], f"{model_config['model_name']}.pth")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=model_config["lr"])

    if os.path.exists(model_file_path): model.load_state_dict(torch.load(model_file_path))

    epochs = model_config["epochs"]

    for epoch in range(epochs): 
        model.train()
        for inputs, targets in trainloader:
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()   

            tgt_mask = torch.triu(torch.ones(model_config["post_days"], model_config["post_days"], device=inputs.device, dtype=torch.bool))

            x = model.encode(inputs,None)
            x = model.decode(x,None,targets,tgt_mask)
            x = model.project(x)

            loss = criterion(x, targets)
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), model_file_path)

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    torch.save(model.state_dict(), model_file_path)

#evaluateModel
def evaluateModel (evalloader :DataLoader ,model: model, model_config: dict):

    model_file_path = os.path.join(model_config["model_folder"], f"{model_config['model_name']}.pth")

    if os.path.exists(model_file_path): model.load_state_dict(torch.load(model_file_path))

    model.eval()
    total_loss = 0

    count = 0
    with torch.no_grad():
        for inputs, targets in evalloader:
            inputs, targets = inputs.cuda(), targets.cuda()

            tgt_mask = torch.triu(torch.ones(model_config["post_days"], model_config["post_days"], device=inputs.device, dtype=torch.bool))
            criterion = nn.MSELoss()
            x = model.encode(inputs,None)
            x = model.decode(x,None,targets,tgt_mask)
            x = model.project(x)

            
            loss = criterion(x, targets)
            total_loss += loss.item()
            count += 1

    average_loss = total_loss / count
    print(f'databatch Loss: {average_loss:.4f}')
    return average_loss

# train multiple datasets
def trainMultipleDatsets(model: model, model_config: dict, dataset_epoch: int,
                         dataset_folder:str, dataset_number:int, batch_size : int, shuffle = True):
    for i in range (dataset_epoch):
        print (f"dataset picked {i}")
        loader = getDataLoader (dataset_folder = dataset_folder, dataset_number = dataset_number, batch_size = batch_size, shuffle = shuffle)
        trainModel(loader,model,model_config)

# evalute multiple datasets
def evaluateMultipleDatsets(model: model, model_config: dict, dataset_epoch: int,
                         dataset_folder:str, dataset_number:int, batch_size : int, shuffle = True):
    count = 0
    total_loss = 0.0
    for i in range (dataset_epoch):
        print (f"dataset picked {i}")
        loader = getDataLoader (dataset_folder = dataset_folder, dataset_number = dataset_number, batch_size = batch_size, shuffle = shuffle)
        total_loss += evaluateModel(loader,model,model_config)
        count += 1
    average_loss = total_loss/count
    print(f'total Loss: {average_loss:.4f}')