import model
from model import Encoder,Decoder,InputEmbeddings,PositionalEncoding,ProjectionLayer,FeedForwardBlock, build_stock_transformer
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import config
from pprint import pprint
import numpy as np
import os

Config = config.get_config()
akConfig = config.get_ak_config()
modelConfig = config.get_transformer_model_config()


def trainModel (train_loader,model, general = False):
    
    if (general == False):
        model_file_path = os.path.join(Config["transformer_model_folder"], f"{akConfig['specific_stock_name']}.pth")
    else:
        model_file_path = os.path.join(Config["general_transformer_model_folder"], f"{Config['generalTransformerName']}.pth")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config["lr"])


    if os.path.exists(model_file_path): model.load_state_dict(torch.load(model_file_path))

    if (general==False):
        epochs = Config["num_epochs"]
    else:
        epochs = modelConfig["epochs"]
    
    for epoch in range(epochs): 
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()   

            tgt_mask = torch.triu(torch.ones(Config["post_days"], Config["post_days"], device=inputs.device, dtype=torch.bool))

            x = model.encode(inputs,None)
            x = model.decode(x,None,targets,tgt_mask)
            x = model.project(x)

            loss = criterion(x, targets)
            loss.backward()
            optimizer.step()
        
        if (epoch) % 20 == 0:
            torch.save(model.state_dict(), model_file_path)
        # save every epoch is general = true
        if (general == True):
            torch.save(model.state_dict(), model_file_path)

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    torch.save(model.state_dict(), model_file_path)

def evaluateModel (test_loader,model):


    model_file_path = os.path.join(Config["transformer_model_folder"], f"{akConfig['specific_stock_name']}.pth")

    if os.path.exists(model_file_path): model.load_state_dict(torch.load(model_file_path))

    model.eval()
    test_loss = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            tgt_mask = torch.triu(torch.ones(Config["post_days"], Config["post_days"], device=inputs.device, dtype=torch.bool))
            criterion = nn.MSELoss()
            x = model.encode(inputs,None)
            x = model.decode(x,None,targets,tgt_mask)
            x = model.project(x)

            
            loss = criterion(x, targets)
            test_loss += loss.item()

    average_loss = test_loss / len(test_loader)
    print(f'Test Loss: {average_loss:.4f}')