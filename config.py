from pathlib import Path

def get_config():
    return {
        "src_features":10,
        "tgt_features" : 2,

        "prev_days": 90,
        # Actual predicted days is one less than post days, since one day is used as SOS token :(
        # Might fix later
        "post_days": 5,

        "d_model": 32,
        "num_heads":8,
        "encoder_layer":6,
        "decoder_lyaer":4,

        "batch_size": 4,
        "num_epochs": 100,
        "lr": 10**-3,
        
        "transformer_model_folder": "./model_weights/StockTransformer",
    }

def get_ak_config():

    return {

        "specific_stock_name":"南华生物",

        "stockFolder":'./akshare_data/dailyStock',

        "symbol_to_name":'./akshare_data/symbol_to_name.json',
        "name_to_symbol":'./akshare_data/name_to_symbol.json',

        "date_start" : "20100101",
        "date_end" :"20200101",

        "train_test_split" : 0.5,

        "generalFolder":'./akshare_data/AllStockData',
    }

def get_visual_config():
    return{
        "visualFolder":"./prediction_eval",
    }
