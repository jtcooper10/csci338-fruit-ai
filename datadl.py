import fiftyone.zoo as fz


# output_dir = r"C:\Path\To\Save\Your\Data\To"
output_dir = r"D:\Projects\Data\FruitAI"
sample_size = 100
labels = ["Apple", "Banana", "Lemon"]


def load(split="validate"):
    training_set = fz.load_zoo_dataset(
        "open-images-v6",
        split=split,
        label_types=["detections"],
        classes=labels,
        max_samples=sample_size,
        dataset_dir=output_dir,
    )
    return training_set


if __name__ == "__main__":
    train = load(split="train")
    print(train)
