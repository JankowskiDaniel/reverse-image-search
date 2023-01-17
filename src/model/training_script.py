import itertools

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import cuda, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertTokenizer

from model import ClipBasedModel
from settings import CAPTIONS_PATH
from utils import EmbeddingDataset, transform

EPOCHS = 4
BATCH_SIZE = 16

captions_df = pd.read_csv(CAPTIONS_PATH)

training_df, valid_df = train_test_split(
    captions_df, test_size=0.2, random_state=12, shuffle=12
)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Define pytorch dataset
training_dataset = EmbeddingDataset(
    training_df, tokenizer=tokenizer, transform=transform
)
validation_dataset = EmbeddingDataset(
    valid_df, tokenizer=tokenizer, transform=transform
)

# Load datasets into torch dataloaders
training_loader = DataLoader(
    training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
validation_loader = DataLoader(
    validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

device = "cuda" if cuda.is_available else "cpu"

model = ClipBasedModel().to(device)

params = [
    {"params": model.resnet_encoder.parameters(), "lr": 1e-4},
    {"params": model.bert_encoder.parameters(), "lr": 1e-5},
    {
        "params": itertools.chain(
            model.image_mapper.parameters(), model.text_mapper.parameters()
        ),
        "lr": 1e-3,
        "weight_decay": 1e-3,
    },
]

optimizer = torch.optim.AdamW(params, weight_decay=0.0)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=1, factor=0.8
)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer,
    scheduler,
):
    min_valid_loss = np.inf
    history = {"train_loss": [], "valid_loss": []}
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_batches_loss = []
        for _, data in enumerate(train_loader, 0):
            model.zero_grad()
            optimizer.zero_grad()
            data = {key: tensor.to(device) for key, tensor in data.items()}

            loss = model(data)
            loss.backward()
            optimizer.step()
            train_batches_loss.append(loss.item())
        train_loss = np.sum(train_batches_loss) / len(train_batches_loss)
        history["train_loss"].append(train_loss)

        model.eval()
        valid_batch_loss = []
        with torch.no_grad():
            for _, data in enumerate(valid_loader, 0):
                data = {key: tensor.to(device) for key, tensor in data.items()}
                loss = model(data)
                valid_batch_loss.append(loss.item())
            valid_loss = np.sum(valid_batch_loss) / len(valid_batch_loss)
            history["valid_loss"].append(valid_loss)

        if min_valid_loss > valid_loss:
            torch.save(model.state_dict(), "best_model.pth")
            min_valid_loss = valid_loss

        print(
            f"Epoch {epoch} \t\t Training Loss: {train_loss} \t\t Validation Loss: {valid_loss}"
        )
        scheduler.step(valid_loss)

    torch.save(model.state_dict(), "model.pth")
