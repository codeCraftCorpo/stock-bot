from pathlib import Path
import os
import torch


#standardize this format across all models
def get_transformer_model_config():
    return {
        #data
        "prev_days": 90,
        "post_days": 5,

        #model
        "src_features":10,
        "tgt_features" : 2,

        "d_model":64,
        "heads":8,
        "d_ff" : 128,
        "dropout" : 0.1,

        #encoder and decoder numbers
        "N": 6,

        #training
        "dataset_epoch": 300,
        "dataset_number": 200,
        "batch_size" :128,
        "epochs" : 10,
        "lr": 10**-3,

        #folders and names
        "model_folder":"./model_weights/transformer",
        "model_name":"transformer_model",

        #Sos token, shape (1, tgt_feature)
        "sosToken" : torch.full((1, 2), -1, dtype=torch.float).cuda()
    }

#static config. can create at the beginning of file
def get_ak_config():

    return {
        #specific stock
        "specific_stock_name":"鼎龙股份",

        #dictionaries for symbol name conversion
        "symbol_to_name":'./akshare_data/symbol_to_name.json',
        "name_to_symbol":'./akshare_data/name_to_symbol.json',
        "idx_to_dataset":'./akshare_data/idx_to_dataset.json',

        #date period for train and test
        "date_start" : "20100101",
        "date_end" :"20200101",

        "eval_data_start" :"20200101",
        "eval_data_end" :"20220101",

        #folders
        "all_stock_data":'./akshare_data/all_stock_data',
        "eval_all_stock_data":'./akshare_data/eval_all_stock_data',
        "all_stock_dataset":'./akshare_data/all_stock_dataset',
        "eval_all_stock_dataset":'./akshare_data/eval_all_stock_dataset',
        "pred_all_stock_folder": './akshare_data/pred_all_stock'   
    }

#returns configs
def getConfigs():
    ak_config = get_ak_config()
    transformer_config = get_transformer_model_config()

    return ak_config,transformer_config


# makes the empty folders
def makeFolders():
    transformerModelConfig = get_transformer_model_config()
    akConfig = get_ak_config()
    os.makedirs(transformerModelConfig["model_folder"],exist_ok=True)

    os.makedirs(akConfig["all_stock_data"], exist_ok=True)
    os.makedirs(akConfig["eval_all_stock_data"], exist_ok=True)
    os.makedirs(akConfig["all_stock_dataset"],exist_ok=True)
    os.makedirs(akConfig["eval_all_stock_dataset"],exist_ok=True)
    os.makedirs(akConfig["pred_all_stock_folder"],exist_ok=True)

makeFolders()                                           