from pathlib import Path
import os

def get_transformer_model_config():
    return {
        "encoder_numbers": 6,
        "decoder_numbers": 6,
        "d_model":64,
        "heads":8,
        "d_ff" : 128,

        "pickDatasetEpoch": 100,
        "dataset_number": 50,
        "batch_size" :128,
        "epochs" : 10,
    }

def get_config():
    return {
        "src_features":10,
        "tgt_features" : 2,

        "prev_days": 90,
        # Actual predicted days is one less than post days, since one day is used as SOS token :(
        # Might fix later
        "post_days": 11,

        "batch_size": 4,
        "num_epochs": 1000,
        "lr": 10**-3,
        
        "transformer_model_folder": "./model_weights/StockTransformer",

        # GENERAL MODEL
        "general_transformer_model_folder":"./model_weights/GeneralTransformer",

        "generalTransformerName":"general_transformer_model"
    }

def get_ak_config():

    return {
        "specific_stock_name":"安徽建工",

        "stockFolder":'./akshare_data/dailyStock',

        "symbol_to_name":'./akshare_data/symbol_to_name.json',
        "name_to_symbol":'./akshare_data/name_to_symbol.json',
        "idx_to_dataset":'./akshare_data/idx_to_dataset.json',

        "date_start" : "20100101",
        "date_end" :"20200101",

        "train_test_split" : 0.5,

        "generalFolder":'./akshare_data/AllStockData',
        "allStockDataset":'./akshare_data/AllStockDataset',
    }

def get_visual_config():
    return{
        "visualFolder":"./prediction_eval",
    }

# makes the empty folders
def makeFolders():
    Config = get_config()
    akConfig = get_ak_config()
    modelConfig = get_transformer_model_config()
    visualConfig = get_visual_config()

    if not os.path.exists(Config["transformer_model_folder"]):
        os.makedirs(Config["transformer_model_folder"])
    if not os.path.exists(Config["general_transformer_model_folder"]):
        os.makedirs(Config["general_transformer_model_folder"])

    if not os.path.exists(akConfig["stockFolder"]):
        os.makedirs(akConfig["stockFolder"])
    if not os.path.exists(akConfig["generalFolder"]):
        os.makedirs(akConfig["generalFolder"])
    if not os.path.exists (akConfig["allStockDataset"]):
        os.mkdirs(akConfig["allStockDataset"])
    
    if not os.path.exists(visualConfig["visualFolder"]):
        os.makedirs(visualConfig["visualFolder"])


makeFolders()