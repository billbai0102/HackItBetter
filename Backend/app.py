from flask import Flask
from flask import request
from flask_ngrok import run_with_ngrok

import alb
import torch
import process
import numpy as np
from torch import nn
from model import UNet
import image_handler as handler

app = Flask(__name__)
run_with_ngrok(app)
model = None

@app.route('/get-output')
def get_output():
    image_ID = request.args.get('image-id')
    image = handler.get_image_by_id(image_ID)
    image = process.preprocess_mri(image)
    mask = model(image)
    mask = process.postprocess_mask(mask)
    overlay = alb.segmentation_mask(image, mask, inner_fill='lightgreen', outer_fill='white')

    return image, mask, overlay

@app.route('/')
def main():
    return "Server is up and running."


if __name__ == '__main__':
    model = UNet(1)
    model.load_state_dict(torch.load('./trained_weights/weights.pt'))
    model.eval()
    app.run()

