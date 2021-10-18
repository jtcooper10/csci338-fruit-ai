#!/usr/bin/env python

import os
from flask import Flask, render_template, request
from PIL import Image
import errno
import sys

rootDir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, rootDir)
import model.fruit

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(rootDir, "web", "static", "images")

# Attempt to create the folder if it doesn't already exist
try:
    os.mkdir(UPLOAD_FOLDER)
except OSError as err:
    if err.errno != errno.EEXIST:
        raise

ALLOWED_EXTENSION = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

modelPath = os.path.join(rootDir, "model.pth")
trainingPath = os.path.join(rootDir, "training")
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
        results = [(label, score) for label, score in mx.predict(img, limit=False).items()]
        results = sorted(results, key=lambda k: -k[1])
        # extract fruit name and confidence from results
        fruitResults, confidence = results[0]
        confidence = "{:.2f}".format(confidence * 100)
    return render_template('results.html', imgpath=newName, aiFruit=fruitResults, aiCon=confidence)


if __name__ == '__main__':
    app.run()
