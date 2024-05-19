import akshare as ak
import config
import json
import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import pandas as pd
import numpy as np
from pprint import pprint

#Global Configs
Config = config.get_config()
akConfig = config.get_ak_config()

#make folders if folders don't exist
def makeFolders():
    if not os.path.exists(akConfig["stockFolder"]):
        os.makedirs(akConfig["stockFolder"])
makeFolders()

#get stock name to symbol, symbol to name dicts into json
def getAkDicts():


    stock_list = ak.stock_zh_a_spot_em()

    symbol_to_name = stock_list.set_index('代码')['名称'].to_dict()
    name_to_symbol = stock_list.set_index('名称')['代码'].to_dict()

    with open(akConfig["symbol_to_name"], 'w', encoding='utf-8') as f:
        json.dump(symbol_to_name, f, ensure_ascii=False, indent=4)

    with open(akConfig["name_to_symbol"], 'w', encoding='utf-8') as f:
        json.dump(name_to_symbol, f, ensure_ascii=False, indent=4)

#load name and symbol dicts
def loadAkDicts():
    try:
        with open(akConfig["symbol_to_name"], 'r', encoding='utf-8') as f:
            symbol_to_name = json.load(f)
    except FileNotFoundError:
        print("symbol_to_name.json file not found. Fetching and saving data...")
        getAkDicts()
        with open(akConfig["symbol_to_name"], 'r', encoding='utf-8') as f:
            symbol_to_name = json.load(f)

    with open(akConfig["name_to_symbol"], 'r', encoding='utf-8') as f:
        name_to_symbol = json.load(f)

    return name_to_symbol, symbol_to_name

# create dataset for one stock
# 日期    开盘    收盘    最高    最低     成交量   成交额    振幅    涨跌幅   涨跌额    换手率
#         0       1      2       3       4        5         6      7         8       9

def getSpecStockCSV():

    name_to_symbol, symbol_to_name = loadAkDicts()

    specStockName = akConfig["specific_stock_name"]
    specStockSymbol = name_to_symbol[specStockName]


    stock_data = ak.stock_zh_a_hist(symbol=specStockSymbol, period="daily", start_date=akConfig["date_start"], end_date=akConfig["date_end"], adjust="qfq")

    csv_file_path = os.path.join(akConfig["stockFolder"], f"{specStockName}.csv")
    stock_data.to_csv(csv_file_path, index=False)

#creates Dataset from one specific stock

class SpecStockData(Dataset):

    def __init__(self):
        csv_file_path = os.path.join(akConfig["stockFolder"], f"{akConfig['specific_stock_name']}.csv")

        if not os.path.exists(csv_file_path):
            getSpecStockCSV()
            print ("not exist file, creating csv")

        self.data = pd.read_csv(csv_file_path, encoding='ISO-8859-1')

        self.inputs = []
        self.outputs = []

        prev = Config["prev_days"]
        post = Config["post_days"]



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

            # （post,2)
            # first element of post is last element of prev
            output_post_days = []
            for j in range (i+prev-1,i+prev-1+post):
                output_day = self.data.iloc[j, 1:3].tolist()
                output_post_days.append(output_day)

            # (i, prev, col)
            # (i, post, 2)
            self.inputs.append(input_prev_days)
            self.outputs.append(output_post_days)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.float), torch.tensor(self.outputs[idx], dtype=torch.float)

# return training and testing dataloader
def getAkDataLoader():
    data = SpecStockData()
    train_size = int(akConfig["train_test_split"] * len(data))

    train_dataset = Subset(data, range(train_size))
    test_dataset = Subset(data, range(train_size, len(data)))

    train_loader = DataLoader(train_dataset, batch_size=Config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config["batch_size"], shuffle=False)
    return train_loader, test_loader