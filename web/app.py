import os
from flask import Flask, render_template, request
import model.fruit
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as func
import torchvision
import numpy as np
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = './static/images'
ALLOWED_EXTENSION = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Main page for project
# Submission button to process images
@app.route('/')
def main_page():
    return render_template('index.html')


# Secondary page to view results of image submission
@app.route('/get_results', methods=['GET', 'POST'])
def get_results():
    if request.method == 'POST':
        # grab file from form
        file = request.files['imageName']
        # split filename to extract extension
        filename, extension = os.path.splitext(file.filename)
        # save filename as temp with original extension (jpg/jpeg/png/etc.)
        newName = "temp" + extension
        # save file in /static/images folder
        fullName = os.path.join(app.config['UPLOAD_FOLDER'], newName)
        file.save(fullName)
    return render_template('results.html', imgpath=newName)


if __name__ == '__main__':
    app.run()
