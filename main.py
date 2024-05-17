import config
from akshares_getdata import getAkDataLoader
from model import build_stock_transformer
from train_and_evaluate import trainModel,evaluateModel


Config = config.get_config()

train_loader, test_loader = getAkDataLoader()

model = build_stock_transformer(Config["src_features"],Config["tgt_features"],Config["prev_days"],Config["post_days"],Config["d_model"]).cuda()

trainModel(train_loader,model)

evaluateModel(test_loader,model)
