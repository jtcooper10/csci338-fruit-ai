# Carmen AI

*Carmen* is an image classification application used to classify a number of different fruits.
Our current image set contains:

* Apples
* Apricots
* Bananas
* Blueberries
* Cantaloupe
* Cherries
* Cocos
* Guavas
* Kiwis
* Lemons
* Limes
* Mangoes
* Oranges
* Papayas
* Peaches
* Pears
* Pineapples
* Raspberries
* Strawberries
* Tomatoes
* Watermelons

For best results, it is recommended to use cropped images, where the edges of the fruit meets the edge of the image.
This engine currently only supports image classification, and accounting for background area and/or competing objects
falls within the domain of object recognition.

## Usage

To start the webserver, make sure you have the following dependencies installed:

* Flask
* PyTorch
* Torchvision

Then, navigate to the `web/` directory and run:

```shell
flask run
```

Visit the webpage on port 5000 to use.
