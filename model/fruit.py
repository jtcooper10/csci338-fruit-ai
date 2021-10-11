import torch
import torchvision
import torchvision.models.detection as models
from torchvision.datasets import ImageFolder


class FruitModel:
    def __init__(self, model: "torch.Module" = None, threshold=0.6):
        if model is None:
            # Set pretained=False once we have our own training pipeline
            model = models.retinanet_resnet50_fpn(pretrained=True)
        self.__model = model
        # TODO: replace default labels with our own
        self.__labels = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        self.threshold = float(threshold)

    @classmethod
    def get_transform(cls):
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.ToTensor(),
        ])

    @classmethod
    def transform(cls, data):
        return cls.get_transform()(data)

    def predict(self, img_data: "torch.Tensor") -> "list[tuple[str, float]]":
        img_data = torch.unsqueeze(img_data, 0)
        m = self.__model.eval()
        with torch.no_grad():
            predictions = m(img_data)[0]
        return [
            (self.__labels[label], score.item())
            for label, score in zip(predictions["labels"], predictions["scores"])
            if score.item() >= self.threshold
        ]


class FruitTrainingModel:
    def __init__(self, model: "FruitModel"):
        self.__model = model

    def detach(self):
        model = self.__model
        self.__model = None
        return model

    def run_epoch(self):
        pass

    @classmethod
    def get_dataset(cls, root: "str"):
        return ImageFolder(root=root, transform=FruitModel.get_transform())
