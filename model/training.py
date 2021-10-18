import math
import random

import model.fruit
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as func
import torchvision
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
import os

path = r"InputPath"
def train_loop(dataloader, model)->model:
    size = len(dataloader.dataset)
    aver = 0.0
    for batch, data in enumerate(dataloader):
        # Compute prediction and loss
        #pred = m.predict(X)
        loss = m.run_epoch(data)
        #if batch % 100 == 0:
        current = batch * len(data)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return m

def test_loop(pop, m, loss_fn) -> float:
    dataloader,classes = pop
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    test_loss, correct = 0.0, 0.0
    keys = dict((v, k) for k, v in classes.items())
    with torch.no_grad():
        a = 0
        Imgdata = datasets.ImageFolder(path)
        path2, dirs, files = next(os.walk(path + "\\" + keys[0]))
        count = len(files)
        sizecopy = size
        q = 0
        while q < size:
            rand = random.randint(0, sizecopy-1)
            temp = 0
            while temp < rand:
                temp+=count
            temp = math.floor(temp/count)
            newdata = Imgdata.imgs.pop(rand)
            Imgdata.imgs.insert(rand,newdata)
            image = Image.open(newdata[0])
            pred = m.predict(image)
            if len(pred) > 0:
                # test_loss += pred[0]
                if not pred.get(keys[temp-1], -1) == -1: correct += 1
            q += 1
            sizecopy-=1


    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%\n")
    return correct




if __name__ == '__main__':

    batch = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #print('Using {} device'.format(device))
    device = 'cpu'
    #model.to(device)
    if device=='cpu':
            pop = model.fruit.FruitTrainingModel.get_data(root=path,
                                                                    batch_size=batch,
                                                                    shuffle=True,
                                                                    num_workers=4)
            training_set, classes = pop
    else:
        training_set, classes = model.fruit.FruitTrainingModel.get_data(
            root=path,
            batch_size=batch,
            shuffle=True,
            pin_memory=True)
    classes = sorted(list(classes.keys()), key=lambda cls: classes[cls])

    m = model.fruit.FruitTrainingModel(labels=classes)
    num_epochs = 8
    best_acc = 0.0
    loss = []
    loss_fn = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct =0


    try:
        print("Training model...")
        for t in range(num_epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            m = train_loop(training_set, m)
            acc = test_loop(pop, m, loss_fn)
            if acc >= best_acc:
                m.export(r"Output path")
                best_accuracy = acc

    except KeyboardInterrupt:
        print("Training was interrupted")



