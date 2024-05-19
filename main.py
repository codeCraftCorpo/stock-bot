import config
from akshares_getdata import getAkDataLoader, SpecStockData,getAllStocksCSV
from model import build_stock_transformer
from train_and_evaluate import trainModel,evaluateModel
from visualization import writePredCsv, visualize
from pprint import pprint
import os
import torch

'''
Config = config.get_config()

train_loader, test_loader = getAkDataLoader()

model = build_stock_transformer(Config["src_features"],Config["tgt_features"],Config["prev_days"],Config["post_days"],Config["d_model"]).cuda()

trainModel(train_loader,model)

evaluateModel(test_loader,model)

writePredCsv()

visualize()
'''
getAllStocksCSV()