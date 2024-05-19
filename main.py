import config
import akshares_getdata
from akshares_getdata import getAkDataLoader, randomDatasetPicker
from model import generalTransformerBuild
from train_and_evaluate import trainModel,evaluateModel
from visualization import writePredCsv, visualize
from pprint import pprint
import os
import torch
from torch.utils.data import DataLoader, Dataset

Config = config.get_config()

akConfig = config.get_ak_config()

modelConfig = config.get_transformer_model_config()

'''
# specific stock model code
train_loader, test_loader = getAkDataLoader()

model = build_stock_transformer(Config["src_features"],Config["tgt_features"],Config["prev_days"],Config["post_days"],Config["d_model"]).cuda()

trainModel(train_loader,model)

evaluateModel(test_loader,model)

writePredCsv()

visualize()
'''

'''
model = generalTransformerBuild()

for i in range (modelConfig["pickDatasetEpoch"]):
    print (f"dataset picked {i}")
    loader = randomDatasetPicker()
    trainModel(loader,model,True)
'''
visualize(True)