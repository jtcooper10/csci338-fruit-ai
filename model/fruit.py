import torch
import torch.nn.functional
import torch.optim as opt
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
from typing import Union


class FruitModel:
    def __init__(self, labels: "list[str]", model: "torch.nn.Module" = None, threshold=0.6):
        """
        Wrapper class for PyTorch model prediction.
        The model may additionally be instantiated from a .pth file using from_file().
        Once instantiated, it may be used for classifying images.

        :param labels: List of indexed label strings representing the names for each classification.
        If providing a pre-built model, the labels must match those that the model was trained with.
        :type labels: list[str]

        :param model: Optional PyTorch module to predict/train against.
        The project default model will be created automatically if not provided.
        In most scenarios, there is no need to provide a model (unless you are experimenting with different ResNets).
        (If you are wanting to use a pre-existing model without experimenting, use from_file() instead)
        :type model: torch.Module

        :param threshold: Value on range (0.0, 1.0) where only predictions which meet the confidence threshold
        are considered valid. Any predictions whose confidence scores do not exceed the threshold are
        deemed invalid and are discarded.
        :type threshold: float

        :raises ValueError: When the number of labels provided does not match the number of model features.
        """
        self._labels = labels

        if model is None:
            model = torchvision.models.resnet50(pretrained=True)
            model.fc = torch.nn.Linear(model.fc.in_features, len(self._labels))
        elif model.fc.out_features != len(labels):
            raise ValueError(f"The PyTorch module provided ({model.fc.out_features} features) "
                             f"does not match the labels provided ({len(labels)} labels)")
        self._model = model.to("cpu")

        self.threshold = float(threshold)

    @classmethod
    def get_transform(cls) -> "torchvision.transforms.Compose":
        """
        Generate the default PyTorch transformation for converting image data to a normalized tensor.

        :return: Instance of default Torchvision transformation object
        """
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.ToTensor(),
        ])

    @classmethod
    def transform(cls, data: "Image.Image") -> "torch.tensor":
        """
        Generate the default PyTorch transformation and apply it against the given image data.
        Useful for one-off transformations, but if repeated transformations are needed,
        it is preferable to use get_transform to get a reusable transform function.

        :param data: Image data to apply the default transformation against.
        :type data: PIL.Image.Image

        :return: Transformed and normalized tensor data based on the provided image.
        """
        return cls.get_transform()(data)

    @torch.no_grad()
    def predict(self, img_data: "Union[torch.Tensor, Image.Image]", limit=True) -> "dict[str, float]":
        """
        Determine what classifications the given image belongs to and their probabilities.
        Either a pre-processed Tensor or PIL image may be provided.

        :param img_data: Image to generate predictions for.
        A PyTorch tensor is greatly preferred, but a PIL image can use default transformations to pre-process for you.
        An example of this usage:
        >>> from PIL import Image
        >>> model = FruitModel(labels["example1", "example2"])
        >>> model.predict(Image.open(r"/path/to/image"))
        <<< dict({ "example1": 0.7 })

        :param limit: Indicate whether or not low-scoring predictions should be filtered out.
        If set to False, then all possible labels will be returned with their corresponding confidence scores.
        :type limit: bool

        :return: Dictionary containing all confident classifications and their confidence scores.
        Any classifications whose confidence scores are below the threshold are filtered out, unless limit=False.
        """
        if isinstance(img_data, Image.Image):
            img_data = FruitModel.transform(img_data).unsqueeze(0)

        m = self._model.eval()
        predictions = m(img_data)
        probabilities = torch.nn.functional.softmax(predictions, dim=1).squeeze(0)
        return {
            self._labels[label_id]: score.item()
            for label_id, score in enumerate(probabilities)
            # Filtering only applied if limit=True.
            # When limit=False, the condition below is always true and so no items are removed.
            if not limit or score.item() >= self.threshold
        }

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
    def __init__(self, model: "torch.nn.Module" = None, threshold=0.6, labels: "list[str]" = None,
                 optimizer=None, loss=None):
        """
        Helper class for model training.
        Extends from the base FruitModel, and as such may be used for predictions as well as training.

        :param model: Optional pre-existing PyTorch module.
        Only recommended when experimenting with different PyTorch backbones.
        :type model: torch.nn.Module

        :param threshold: Cutoff value for an "accurate" prediction.
        Lower threshold values result in more false positives, while higher values result in more missed predictions.
        :type threshold: float

        :param labels: Ordered list of classification labels for the model.
        Indexed by classification ID, i.e. labels[0] should return the label of classification 0.
        :type labels: list[str]

        :param optimizer: Optional PyTorch optimizer.
        Only recommended when experimenting with different PyTorch optimizers.
        :param loss: Optional PyTorch loss function.
        Only recommended when experimenting with different PyTorch loss functions.
        """
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
        m = self._model.train()
        values, labels = data
        self.__optimizer.zero_grad()
        result = m(values)
        loss = self.__loss(result, labels)
        loss.backward()
        self.__optimizer.step()
        return loss.item()

    @classmethod
    def get_data(cls, root: "str",
                 transform: "torchvision.transforms.Compose" = None,
                 **loader_options) -> "tuple[DataLoader, dict[str, int]]":
        """
        Dataset import helper using PIL images for training/testing purposes.
        The classifications of each dataset must match during training, testing, and inference.

        :param root: Root directory where image classification folders are kept.
        Each sub-folder in the root directory should contain a single image category.
        The classification names are derived from the name of each folder.
        (It is highly recommended to use the EXACT same root folder structure for each model version)
        :type root: str

        :param transform: Optional torchvision transform.
        If not specified, the default transformer is used.
        (Option is only useful if experimenting with different transforms)
        :type transform: torchvision.transforms.Compose

        :param loader_options: Dictionary containing options to pass to the dataloader.

        :return: Pair of type (DataLoader, dict) where the dict maps label names to their corresponding ids.
        """
        if transform is None:
            transform = FruitModel.get_transform()
        img_set = ImageFolder(root=root, transform=transform)
        return DataLoader(dataset=img_set, **loader_options), img_set.class_to_idx
