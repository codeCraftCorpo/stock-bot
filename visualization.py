import config
import os
import pandas as pd
from model import build_stock_transformer
from akshares_getdata import SpecStockData
from torch.utils.data import DataLoader
import torch
import numpy as np
from pprint import pprint
import pyglet
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *



#get configs, make folder, get read and write file paths
akConfig = config.get_ak_config()
Config = config.get_config()
vConfig = config.get_visual_config()

if not os.path.exists(vConfig["visualFolder"]):
    os.makedirs(vConfig["visualFolder"])

writeFilePath = os.path.join(vConfig["visualFolder"],f"{akConfig["specific_stock_name"]}.csv")
model_file_path = os.path.join(Config["transformer_model_folder"], f"{akConfig['specific_stock_name']}.pth")

# save predicted prices of a specific stock to csv
# PRED open, PRED close, Actual open, Actual close
# 0          1           2            3

def writePredCsv():
    #get dataloader and set up model
    data = SpecStockData()
    loader = DataLoader(data, 1, shuffle=False)

    model = build_stock_transformer(Config["src_features"],Config["tgt_features"],Config["prev_days"],Config["post_days"],Config["d_model"])
    if os.path.exists(model_file_path): model.load_state_dict(torch.load(model_file_path))
    model.cuda()
    model.eval()


    #gets prediction, actual data into a tensor
    pred = torch.tensor([], dtype=torch.float32).cuda()
    actual = torch.tensor([], dtype = torch.float32).cuda()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            tgt_mask = torch.triu(torch.ones(Config["post_days"], Config["post_days"], device=inputs.device, dtype=torch.bool))
            x = model.encode(inputs,None)
            x = model.decode(x,None,targets,tgt_mask)
            result = model.project(x)

            pred = torch.cat((pred,result[0]), dim = 0)
            actual = torch.cat((actual,targets[0]),dim = 0)

    #write to csv
    #Actual open, actual close, pred open, pred close
    pred_np = pred.cpu().numpy()
    actual_np = actual.cpu().numpy()
    combined = np.hstack((actual_np, pred_np))
    df = pd.DataFrame(combined, columns=["actual open", "actual close", "pred open", "pred close"])
    df.to_csv(writeFilePath, index=False)


# visualize with pyglet
def visualize():
    data = pd.read_csv(writeFilePath)
    actual_open = data.iloc[:, 0].values
    pred_open = data.iloc[:, 2].values
    time_index = 0  # Starting time index

    # Constants
    WINDOW_WIDTH = 1600
    WINDOW_HEIGHT = 900
    DAYS_STEP = 10
    VIEWPORT_DAYS = 50  # Number of days visible in the viewport

    # Pyglet Window
    window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT, "Actual vs Predicted Open Prices")

    def create_lines_and_labels(time_index):
        end_index = min(time_index + VIEWPORT_DAYS, len(actual_open))
        actual_slice = actual_open[time_index:end_index]
        pred_slice = pred_open[time_index:end_index]
        
        max_val = max(max(actual_slice), max(pred_slice))
        min_val = min(min(actual_slice), min(pred_slice))
        y_scale = WINDOW_HEIGHT / (max_val - min_val)
        x_scale = WINDOW_WIDTH / (end_index - time_index)
        
        actual_lines = []
        pred_lines = []
        y_labels = []
        x_labels = []
        #
        for i in range(len(actual_slice) - 1):
            actual_lines.append(pyglet.shapes.Line(x_scale * i, y_scale * (actual_slice[i] - min_val), 
                                                   x_scale * (i + 1), y_scale * (actual_slice[i + 1] - min_val), 
                                                   width=2, color=(34, 139, 34)))
            pred_lines.append(pyglet.shapes.Line(x_scale * i, y_scale * (pred_slice[i] - min_val), 
                                                 x_scale * (i + 1), y_scale * (pred_slice[i + 1] - min_val), 
                                                 width=2, color=(218, 112, 214)))

        # Create y-axis labels
        y_step = (max_val - min_val) / 10
        for i in range(11):
            y_value = min_val + i * y_step
            y_labels.append((0, y_scale * (y_value - min_val), f"{y_value:.2f}"))

        # Create x-axis labels
        for i in range(time_index, end_index, DAYS_STEP):
            x_labels.append((x_scale * (i - time_index), 0, f"Day {i + 1}"))
        
        return actual_lines, pred_lines, y_labels, x_labels

    actual_lines, pred_lines, y_labels, x_labels = create_lines_and_labels(time_index)

    @window.event
    def on_draw():
        window.clear()
        for line in actual_lines:
            line.draw()
        for line in pred_lines:
            line.draw()

        for x, y, text in y_labels:
            label = pyglet.text.Label(text, x=x + 5, y=y, anchor_x='left', anchor_y='center')
            label.draw()

        for x, y, text in x_labels:
            label = pyglet.text.Label(text, x=x, y=y + 5, anchor_x='center', anchor_y='bottom')
            label.draw()

    @window.event
    def on_key_press(symbol, modifiers):
        nonlocal time_index, actual_lines, pred_lines, y_labels, x_labels
        if symbol == pyglet.window.key.RIGHT:
            time_index = min(time_index + DAYS_STEP, len(actual_open) - VIEWPORT_DAYS)
        elif symbol == pyglet.window.key.LEFT:
            time_index = max(time_index - DAYS_STEP, 0)

        # Update lines and labels based on the new time_index
        actual_lines, pred_lines, y_labels, x_labels = create_lines_and_labels(time_index)

    pyglet.app.run()