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
        self.tokenizer = tokenizer
        self.transform = transform

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        caption: str = self.captions[idx]
        tokens_caption: Dict[str, List[int]] = self.tokenizer.encode_plus(
            caption,
            None,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )

        image = cv2.imread(IMAGES_PATH + self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)

        ids = tokens_caption["input_ids"]
        mask = tokens_caption["attention_mask"]
        token_type_ids = tokens_caption["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "image": image,
        }

    def __len__(self):
        return len(self.captions)


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class EmbeddingMapper(nn.Module):
    def __init__(
        self, embedding_dim: int, project_dim: int = 256, dropout: float = 0.2
    ) -> None:
        super(EmbeddingMapper, self).__init__()
        self.l1 = nn.Linear(embedding_dim, project_dim)
        self.gelu = nn.GELU()
        self.l2 = nn.Linear(project_dim, project_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(project_dim)

    def forward(self, embedding):
        output = self.l1(embedding)
        embedding = self.gelu(output)
        embedding = self.l2(embedding)
        embedding = self.dropout(embedding)
        embedding = embedding + output
        embedding = self.norm(embedding)
        return embedding
