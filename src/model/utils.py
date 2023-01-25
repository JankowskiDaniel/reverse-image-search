from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import DistilBertTokenizer

from settings import CAPTIONS_PATH, IMAGES_PATH


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        captions: pd.DataFrame,
        tokenizer: DistilBertTokenizer,
        transform: transforms.Compose,
    ) -> None:
        """Class for processing input texts and images. Transform both into proper form and torch tensors.

        Args:
            captions (pd.DataFrame): Contains two columns: image filename and description of it
            tokenizer (DistilBertTokenizer): Tokenizer
        """
        self.images = captions["image"].tolist()
        self.captions = captions["caption"].tolist()
        self.encoded_captions = tokenizer(
            self.captions, padding=True, truncation=True, max_length=CFG.max_length, return_token_type_ids=True,
        )
        self.transform = transform

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        caption = self.captions[idx]

        image = cv2.imread("/content/Images/" + self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)

        tokens_caption = {
              key: values[idx]
              for key, values in self.encoded_captions.items()
          }
        ids = tokens_caption["input_ids"]
        mask = tokens_caption["attention_mask"]
        token_type_ids = tokens_caption["token_type_ids"]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "image": image,
            "caption": caption
        }


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

