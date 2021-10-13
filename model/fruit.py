import torch
import torch.nn.functional
import torch.optim as opt
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class FruitModel:
    def __init__(self, labels: "list[str]", model: "torch.nn.Module" = None, threshold=0.6):
        """
        Wrapper class for PyTorch model prediction and training.

        :param model: PyTorch module to predict/train against.
        :type model: torch.Module

        :param threshold: Value on range (0.0, 1.0] where only predictions which meet the confidence threshold
        are considered valid. Any predictions whose confidence scores do not exceed the threshold are
        deemed invalid and are discarded.
        :type threshold: float

        :param labels: Optional list of indexed label strings representing the names for each classification.
        :type labels: list[str]
        """
        self._labels = labels

        if model is None:
            model = torchvision.models.resnet50(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, len(self._labels))
        self._model = model.to("cpu")

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
        m = self._model.eval()
        with torch.no_grad():
            predictions = m(img_data)
        probabilities = torch.nn.functional.softmax(predictions, dim=1).squeeze(0)
        return [
            (self._labels[label_id], score.item())
            for label_id, score in enumerate(probabilities)
            if score.item() > self.threshold
        ]

    @classmethod
    def from_file(cls, path: "str", labels: "list[str]", threshold=0.6) -> "FruitModel":
        """
        Import a model from a .pth file.

        :param path: Absolute location of file to import.
        :type path: str

        :param threshold: Value on range (0.0, 1.0] where only predictions which meet the confidence threshold
        are considered valid. Any predictions whose confidence scores do not exceed the threshold are
        deemed invalid and are discarded.
        :type threshold: float

        :param labels: Optional list of indexed label strings representing the names for each classification.
        :type labels: list[str]

        :return: Model instantiated from the indicated .pth file.
        """
        pth_model = torchvision.models.resnet50(pretrained=False, num_classes=len(labels))
        pth_model.load_state_dict(torch.load(path))
        return FruitModel(model=pth_model, threshold=threshold, labels=labels)

    def export(self, path: "str"):
        """
        Save the given model as a .pth file.

        :param path: Absolute location of file to export to.
        :type path: str
        """
        torch.save(self._model.state_dict(), path)


class FruitTrainingModel(FruitModel):
    def __init__(self, model: "torch.nn.Module" = None, threshold=0.6, labels: "list[str]" = None, optimizer=None, loss=None):
        super(FruitTrainingModel, self).__init__(labels=labels, model=model, threshold=threshold)
        self._model.train()

        if optimizer is None:
            optimizer = opt.SGD(self._model.parameters(), lr=0.001)
        self.__optimizer = optimizer

        if loss is None:
            loss = torch.nn.CrossEntropyLoss()
        self.__loss = loss

    def detach(self) -> "FruitModel":
        """
        Generate a new trained model suitable for predictions from the current model.
        All training methods will be disabled in the generated model, and is optimized for inference.

        :return: Non-trainable copy of the training model.
        """
        return FruitModel(labels=self._labels, model=self._model.eval(), threshold=self.threshold)

    def run_epoch(self, data: "Dataset") -> "float":
        """
        Performs a single training iteration on the model with the given PyTorch dataset.
        The labels provided by the dataset MUST match the instantiated model.

        :param data: PyTorch dataset with labels matching the model.
        :type data: torch.utils.data.Dataset

        :return: Float representing loss value of the iteration.
        """
        values, labels = data
        self.__optimizer.zero_grad()
        result = self._model(values)
        loss = self.__loss(result, labels)
        loss.backward()
        self.__optimizer.step()
        return loss.item()

    @classmethod
    def get_dataset(cls, root: "str") -> "ImageFolder":
        return ImageFolder(root=root, transform=FruitModel.get_transform())
