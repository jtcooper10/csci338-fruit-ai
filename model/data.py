import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from fiftyone.core.sample import Sample
import fiftyone.core.dataset
import fiftyone.zoo as fz
from model.fruit import FruitModel


class FiftyOneDataset(Dataset):
    def __init__(self, data: "fiftyone.core.dataset.Dataset", labels: "dict[str, int]", transform):
        """
        Custom PyTorch dataset for the FiftyOne data source.
        Can be iterated over using a standard dataloader.

        :param data: FiftyOne core Dataset object
        :param labels: Map from label names into their label ID
        :param transform: PyTorch transform function to process a PIL image into a corresponding tensor
        """
        self.__data = data
        self.__labels = labels
        self.__samples: "list[str]" = []
        # Samples must be accessed by sample id, so we build an index on sample ids.
        for sample in data:
            self.__samples.append(sample["id"])
        self.__transform = transform

    def __len__(self):
        return len(self.__samples)

    def __getitem__(self, idx):
        sample = self.__data[self.__samples[idx]]
        img_data = Image.open(sample["filepath"])
        img_data = self.__transform(img_data)
        detections = sample["detections"]["detections"]

        def convert_bounds(bounds: "torch.Tensor", width: "float", height: "float"):
            """
            Helper method for converting a FiftyOne bounding box into a PyTorch one,
            while calculating the area of the bounds.

            FiftyOne's bounding boxes use the format: (x, y, width, height), where (x,y) is top left corner
            FiftyOne's (x, y) is on the scale [0, 1]
            PyTorch uses the format: (x1, y1, x2, y2), with bounding corners (x1, y1) and (x2, y2)
            PyTorch's coordinates are pixel-relative (i.e. on the range [img_width, img_height])
            """
            # Rescale the tensor to match the image width and height
            bounds[:, 0::2] *= width
            bounds[:, 1::2] *= height
            calc_areas = torch.prod(bounds[:, 2:], dim=1)
            # To convert (x, y, w, h) -> (x1, y1, x2, y2):
            #   - We keep (x, y) as our (x1, y1)
            #   - We treat (x2, y2) as being to the lower left of (x1, y1)
            #   - x2 is <width> pixels to the right of (x1, y1)
            #   - y2 is <height> pixels DOWN from (x1, y1), do this by making <height> negative
            bounds[:, 3] *= -1
            bounds[:, 2:] += bounds[:, 0:2]
            return bounds, calc_areas

        # Filter out the detections we don't care about (i.e. those which have unwanted labels)
        detections = filter(lambda det: det["label"] in self.__labels, detections)

        # FiftyOne's bounding boxes use the format: (x, y, width, height), where (x,y) is top left corner
        #   FiftyOne's (x, y) is on the scale [0, 1]
        # PyTorch uses the format: (x1, y1, x2, y2), with bounding corners (x1, y1) and (x2, y2)
        #   PyTorch's coordinates are pixel-relative
        boxes, area = convert_bounds(torch.tensor([det["bounding_box"] for det in detections], dtype=torch.float64),
                                     width=img_data.shape[1],
                                     height=img_data.shape[2])
        labels = torch.tensor([self.__labels[det["label"]] for det in detections], dtype=torch.int64)
        # Do we need to flag groups? Idk how this works tbh. For now the answer is always "no" (aka 0)
        crowds = torch.zeros(boxes.size(0), dtype=torch.uint8)
        targets = {
            "image_id": torch.tensor(idx, dtype=torch.int64),
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": crowds,
        }

        return img_data, targets
