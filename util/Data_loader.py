import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class PlantDataset(Dataset):
    def __init__(self, config, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.config = config
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        image_path = self.dataframe['images'].iloc[idx]
        image = Image.open(image_path)
        labels = self.dataframe.iloc[idx]['labels'].split(' ')
        encoded_labels = encode_label(labels, self.config.LABELS)
        if self.transform:
            image = self.transform(image)
        return image, encoded_labels

def load_df(config):
    train_df = pd.read_csv(config.CSV_TRAIN)
    test_df = pd.read_csv(config.CSV_TEST)
    val_df = pd.read_csv(config.CSV_VAL)

    train_df['images'] =  train_df['images'].apply(lambda x: str(config.TRAIN_DIR) +'/'+ x)
    test_df['images'] =  test_df['images'].apply(lambda x: str(config.TEST_DIR) +'/'+ x)
    val_df['images'] =  val_df['images'].apply(lambda x: str(config.VAL_DIR) +'/'+ x)
    config.TRAIN_SIZE = len(train_df)
    config.TEST_SIZE = len(test_df)
    config.VAL_SIZE = len(val_df)
    return train_df, val_df, test_df, config


def get_transforms(config):
    resize = transforms.Resize(size=(config.INPUT_HEIGHT, config.INPUT_WIDTH))
    make_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    auto_augment = transforms.AutoAugment()

    train_transforms = transforms.Compose([
        auto_augment,
        make_tensor,
        normalize
    ])
    val_transforms = transforms.Compose([
        resize,
        make_tensor,
        normalize
    ])
    return train_transforms, val_transforms


def encode_label(labels, class_list):
    """Encode a list of labels using one-hot encoding.

    Args:
        label: A list of labels to encode.
        class_list: A list of all possible labels. Defaults to DEFAULT_LABELS.

    Returns:
        A tensor representing the one-hot encoding of the input labels.
    """
    # Create a tensor of zeros with the same length as the class list
    # labels to one-hot
    target = torch.zeros(len(class_list))
    for label in labels:
        idx = class_list.index(label)
        target[idx] = 1
    return target


def decode_label(encoded_label, class_list):
    """Decode a one-hot encoded label into its original label(s).
    Args:
        encoded_label: A tensor representing the one-hot encoding of a label.
        class_list: A list of all possible labels. Defaults to DEFAULT_LABELS.

    Returns:
        A list of the decoded label(s).
    """
    # Use a list comprehension to create the decoded list
    decoded = [class_list[i] for i, val in enumerate(encoded_label) if val == 1]
    return decoded
