import akshare as ak
from config import getConfigs
import json
import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset,ConcatDataset
import pandas as pd
import numpy as np
from pprint import pprint
import re
import random

#Global Configs
ak_config,_ = getConfigs()

#make folders if folders don't exist


#get stock name to symbol, symbol to name dicts into json
def getAkDicts():
    stock_list = ak.stock_zh_a_spot_em()

    symbol_to_name = stock_list.set_index('代码')['名称'].to_dict()
    name_to_symbol = stock_list.set_index('名称')['代码'].to_dict()

    with open(ak_config["symbol_to_name"], 'w', encoding='utf-8') as f:
        json.dump(symbol_to_name, f, ensure_ascii=False, indent=4)

    with open(ak_config["name_to_symbol"], 'w', encoding='utf-8') as f:
        json.dump(name_to_symbol, f, ensure_ascii=False, indent=4)

#load name and symbol dicts
def loadAkDicts():
    try:
        with open(ak_config["symbol_to_name"], 'r', encoding='utf-8') as f:
            symbol_to_name = json.load(f)
    except FileNotFoundError:
        print("symbol_to_name.json file not found. Fetching and saving data...")
        getAkDicts()
        with open(ak_config["symbol_to_name"], 'r', encoding='utf-8') as f:
            symbol_to_name = json.load(f)

    with open(ak_config["name_to_symbol"], 'r', encoding='utf-8') as f:
        name_to_symbol = json.load(f)

    return name_to_symbol, symbol_to_name

# 日期    开盘    收盘    最高    最低     成交量   成交额    振幅    涨跌幅   涨跌额    换手率
#          0       1      2       3       4        5         6      7         8       9

#write all stocks data to csv
def getAllStocksCSV(csv_folder_path: str, date_start:str, date_end:str):

    name_to_symbol, symbol_to_name = loadAkDicts()
    stockNames = list(symbol_to_name.values())

    def sanitize_filename(filename):
        return re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    def specificStockCSV(stockName: str):

        specStockSymbol = name_to_symbol[stockName]
        stock_data = ak.stock_zh_a_hist(symbol=specStockSymbol, period="daily", start_date=date_start, end_date=date_end, adjust="qfq")

        #remove special characters in stock name to write to csv.
        #might cause reading problems later due to dismatch from name to keys
        sanitzedName = sanitize_filename(stockName)
        csv_file_path = os.path.join(csv_folder_path, f"{sanitzedName}.csv")
        stock_data.to_csv(csv_file_path, index=False)

    for stock in stockNames:
        print (f"getting stock to csv {stock}")
        specificStockCSV(stock)

# create dataset for specific stock
class SpecStockData(Dataset):

    def __init__(self, stock_name:str, csv_folder_path: str, model_config :dict):
        self.initialized = False

        self.stock_name = stock_name.replace('.csv', '')
        self.csv_folder_path = csv_folder_path
        self.model_config = model_config

        csv_file_path = os.path.join(self.csv_folder_path,f"{self.stock_name}.csv")

        try:
            self.data = pd.read_csv(csv_file_path, encoding='ISO-8859-1')
            self.initialized = True
        except Exception as e:
            return
        
        self.inputs = []
        self.outputs = []
        prev = model_config["prev_days"]
        post = model_config["post_days"]

        for i in range(0,len(self.data) - prev-post, post):
            # (prev, col)
            input_prev_days = []
            for j in range (i,i+prev):
                input_day = self.data.iloc[j,1:].values.tolist()
                input_prev_days.append(input_day)

            #Normalize 成交量 成交额 columns
            input_prev_days = np.array(input_prev_days)

            columns_to_normalize = input_prev_days[:, 4:6]
            means = columns_to_normalize.mean(axis=0)
            stds = columns_to_normalize.std(axis=0)
            normalized_columns = (columns_to_normalize - means) / (stds + 0.0000001)
            input_prev_days[:, 4:6] = normalized_columns
            input_prev_days = input_prev_days.tolist()

            #（post,2)
            # first element of post is last element of prev
            output_post_days = []
            for j in range (i+prev-1,i+prev-1+post):
                output_day = self.data.iloc[j, 1:3].tolist()
                output_post_days.append(output_day)

            # (i, prev, 9)
            # (i, post, 2)
            self.inputs.append(input_prev_days)
            self.outputs.append(output_post_days)
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.float), torch.tensor(self.outputs[idx], dtype=torch.float)

# get all stock dataset
def getAllStockDataset(csv_folder_path: str,dataset_folder_path: str,model_config:dict):
    
    count = 0
    for filename in os.listdir(csv_folder_path):
        assert(filename.endswith('.csv'))
        temp_dataset = SpecStockData(stock_name = filename, csv_folder_path = csv_folder_path, model_config = model_config)
        save_path = os.path.join(dataset_folder_path,f"{filename.replace(".csv","")}.pth")

        #if csv file is empty, do not save the dataset
        if temp_dataset.initialized == True: torch.save(temp_dataset, save_path)
        print (f"{count}saving file{filename} dataset")
        count +=1

#dataloader, generates batch_data random numbers from all datasets numbers

#make a json file mapping number to .pth files
def loadDatasetDict(dataset_folder:str):

    def getDatasetJson ():
        count = 0
        dataset_dict = {}
        for filename in os.listdir(dataset_folder):
            dataset_dict[count] = filename
            count +=1
        with open (ak_config["idx_to_dataset"],'w',encoding='utf-8') as f:
            json.dump (dataset_dict, f ,ensure_ascii=False,indent=4)
    
    try:
        with open(ak_config["idx_to_dataset"], 'r', encoding='utf-8') as f:
            result_dict = json.load(f)
    except FileNotFoundError:
        print("idx_to_dataset.json file not found. Fetching and saving data...")
        getDatasetJson()
        with open(ak_config["idx_to_dataset"], 'r', encoding='utf-8') as f:
            result_dict = json.load(f)
    return result_dict

# returns a dataloader generated from concatenated dataset
# random number generator generates the pth files to use
def getDataLoader (dataset_folder:str, dataset_number:int, batch_size : int, shuffle = True) ->DataLoader:
    dict = loadDatasetDict(dataset_folder)
    numbers_list = list(range(len(dict)))
    id_choices = random.sample(numbers_list,dataset_number)
    dataset_list = []
    dataset_names = [dict[str(i)] for i in id_choices]

    for name in dataset_names:
        file_path = os.path.join(dataset_folder,name)
        # try except statement because eval datasets might differ from train
        try:
            dataset = torch.load(file_path)
            dataset_list.append(dataset)
        except Exception as e:
            #print(f"Evaluating, Failed to load dataset from {file_path}: {e}")
            pass

    combined_dataset = ConcatDataset(dataset_list)
    
    train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader

# get csv files, datasets, dataloader
def get_csv_dataset_dataloader(csv_folder_path:str, dataset_folder_path: str,
                               data_start:int, data_end:int,
                               model_config:dict, shuffle = True,
                               create_csv = False, create_dataset = False, create_loader = True
                               ):
    
    if (create_csv == True):
        getAllStocksCSV(csv_folder_path= csv_folder_path, date_start=data_start, date_end=data_end)
    if (create_dataset == True):
        getAllStockDataset(csv_folder_path = csv_folder_path ,dataset_folder_path= dataset_folder_path, model_config = model_config)
    if (create_loader == True):
         return getDataLoader (dataset_folder = dataset_folder_path, dataset_number = model_config["dataset_number"], batch_size = model_config["batch_size"], shuffle = shuffle)
