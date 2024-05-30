from config import getConfigs
import os
import pandas as pd
from akshares_getdata import SpecStockData
from torch.utils.data import DataLoader
import torch
import numpy as np
from pprint import pprint
import model
import pyglet
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *




#get configs, make folder, get read and write file paths
ak_config, transformer_config = getConfigs()

# save predicted prices of specific stock to csv
# PRED open, PRED close, Actual open, Actual close
# 0          1           2            3

def writePredCsv(eval_folder: str, stock_name:str,  model: model, pred_folder:str, model_config: dict):

    #get dataloader and set up model
    dataset = SpecStockData(stock_name = stock_name, csv_folder_path = eval_folder, model_config = model_config)
    loader = DataLoader(dataset, 1, shuffle=False)

    model.cuda()
    model.eval()


    #gets prediction, actual data into a tensor
    pred = torch.tensor([], dtype=torch.float32).cuda()
    actual = torch.tensor([], dtype = torch.float32).cuda()

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            
            x = model(inputs,model_config["sosToken"],model_config["post_days"])
            pred = torch.cat((pred,x[0]), dim = 0)
            actual = torch.cat((actual,targets[0]),dim = 0)

    #write to csv
    #Actual open, actual close, pred open, pred close
    pred_np = pred.cpu().numpy()
    actual_np = actual.cpu().numpy()
    combined = np.hstack((actual_np, pred_np))
    df = pd.DataFrame(combined, columns=["actual open", "actual close", "pred open", "pred close"])
    write_file_path = os.path.join(pred_folder,f"{stock_name}.csv")
    df.to_csv(write_file_path, index=False)



# visualize with pyglet
#THIS VISUALIZES PREDICTION ON TRAIN SET, NOT JUST TEST SET
def visualize(eval_folder: str, stock_name:str,  model: model, pred_folder:str, model_config: dict):
    writePredCsv (eval_folder = eval_folder , stock_name = stock_name,  model = model, pred_folder = pred_folder, model_config = model_config)
    write_file_path = os.path.join(pred_folder,f"{stock_name}.csv")
    data = pd.read_csv(write_file_path)
    actual_open = data.iloc[:, 0].values
    pred_open = data.iloc[:, 2].values
    time_index = 0  # Starting time index

    # Constants
    WINDOW_WIDTH = 1600
    WINDOW_HEIGHT = 900
    DAYS_STEP = 5
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
            time_index = min(time_index + 5*DAYS_STEP, len(actual_open) - VIEWPORT_DAYS)
        elif symbol == pyglet.window.key.LEFT:
            time_index = max(time_index - 5*DAYS_STEP, 0)

        # Update lines and labels based on the new time_index
        actual_lines, pred_lines, y_labels, x_labels = create_lines_and_labels(time_index)

    pyglet.app.run()