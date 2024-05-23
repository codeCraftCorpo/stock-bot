from pathlib import Path
import os

#standardize this format across all models
def get_transformer_model_config():
    return {
        #data
        "prev_days": 90,
        # Actual predicted days is one less than post days, since one day is used as SOS token :(
        "post_days": 11,

        #model
        "src_features":10,
        "tgt_features" : 2,

        "d_model":64,
        "heads":8,
        "d_ff" : 128,

        "encoder_numbers": 6,
        "decoder_numbers": 6,

        #training
        "dataset_epoch": 100,
        "dataset_number": 200,
        "batch_size" :128,
        "epochs" : 10,
        "lr": 10**-3,

        #folders and names
        "model_folder":"./model_weights/transformer",
        "model_name":"transformer_model",
    }

#static config. can create at the beginning of file
def get_ak_config():

    return {
        #specific stock
        "specific_stock_name":"香江控股",

        #dictionaries for symbol name conversion
        "symbol_to_name":'./akshare_data/symbol_to_name.json',
        "name_to_symbol":'./akshare_data/name_to_symbol.json',
        "idx_to_dataset":'./akshare_data/idx_to_dataset.json',

        #date period for train and test
        "date_start" : "20100101",
        "date_end" :"20200101",

        "eval_data_start" :"20200101",
        "eval_data_end" :"20220101",

        #train test split
        "train_test_split" : 0.7,

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


