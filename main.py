from config import getConfigs
import akshares_getdata
from akshares_getdata import get_csv_dataset_dataloader
from model import generalTransformerBuild
from train_and_evaluate import trainMultipleDatsets,evaluateMultipleDatsets
from visualization import writePredCsv, visualize
from pprint import pprint
import os
import torch
from torch.utils.data import DataLoader, Dataset



def mainExecute (intialize = False, train = False, evaluate = False, visualization = False):
    ak_config,transformer_config = getConfigs()

    model = generalTransformerBuild()

    if (intialize == True):
        get_csv_dataset_dataloader(csv_folder_path = ak_config["all_stock_data"], dataset_folder_path= ak_config["all_stock_dataset"],
                               data_start = ak_config["date_start"], data_end = ak_config["date_end"],
                               model_config = transformer_config, shuffle = True,
                               create_csv = True, create_dataset = True, create_loader = False
                               )

        get_csv_dataset_dataloader(csv_folder_path = ak_config["eval_all_stock_data"], dataset_folder_path= ak_config["eval_all_stock_dataset"],
                               data_start = ak_config["eval_data_start"], data_end = ak_config["eval_data_end"],
                               model_config = transformer_config, shuffle = True,
                               create_csv = True, create_dataset = True, create_loader = False
                               )
    
    if (train == True):
        trainMultipleDatsets(model = model, model_config = transformer_config, dataset_epoch= transformer_config["dataset_epoch"],
                         dataset_folder = ak_config["all_stock_dataset"], dataset_number = transformer_config["dataset_number"],
                         batch_size = transformer_config["batch_size"], shuffle = True)
    
    if (evaluate == True):
        evaluateMultipleDatsets(model = model, model_config = transformer_config, dataset_epoch= transformer_config["dataset_epoch"],
                         dataset_folder = ak_config["eval_all_stock_dataset"], dataset_number = transformer_config["dataset_number"],
                         batch_size = transformer_config["batch_size"], shuffle = True)
    
    if (visualization == True):
        visualize(eval_folder = ak_config["eval_all_stock_data"], stock_name = ak_config["specific_stock_name"],  model= model, 
          pred_folder = ak_config["pred_all_stock_folder"], model_config = transformer_config)
    
mainExecute(train=True)