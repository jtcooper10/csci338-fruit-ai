#!/usr/bin/env python

import os
from flask import Flask, render_template, request
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as func
import torchvision
import numpy as np
from PIL import Image
import os
import sys
sys.path.insert(0, '/mnt/c/Users/Nicholas/PycharmProjects/csci338-fruit-ai/')

import model.fruit

app = Flask(__name__)
UPLOAD_FOLDER = './static/images'
ALLOWED_EXTENSION = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

modelPath = r'/mnt/c/Users/Nicholas/Pictures/archive/fruits-360_dataset/fruits-360/AI_fruit/model.pth'
trainingPath = r'/mnt/c/Users/Nicholas/Pictures/archive/fruits-360_dataset/fruits-360/AI_fruit/training'
classes = ['Apple Red 1', 'Apple Red 2', 'Apple Red 3', 'Apricot', 'Banana', 'Blueberry', 'Cantaloupe 1',
           'Cantaloupe 2', 'Cherry 1', 'Cherry 2', 'Cocos', 'Guava', 'Kiwi', 'Lemon', 'Limes', 'Mango', 'Orange',
           'Papaya', 'Peach', 'Peach 2', 'Peach Flat', 'Pear', 'Pear 2', 'Pineapple', 'Raspberry', 'Strawberry',
           'Strawberry Wedge', 'Tomato 1', 'Tomato 2', 'Tomato 3', 'Watermelon']

mx = model.fruit.FruitModel.from_file(labels=classes, path=modelPath)


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
        img = Image.open(fullName)
        # save results
        results = mx.predict(img)
        # extract fruit name and confidence from results
        fruitResults = str(list(results.keys()))[2:-2].lower()
        confidence = "{:.2f}".format(float(str(list(results.values()))[2:-2]) * 100)
    return render_template('results.html', imgpath=newName, aiFruit=fruitResults, aiCon=confidence)


if __name__ == '__main__':
    app.run()
