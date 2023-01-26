from torch import nn
from transformers import DistilBertModel
from torchvision import models
import torch

from settings import (TOKENIZER_NAME,
                      PROJECTION_DIM)

class DualEncoder(nn.Module):
    def __init__(
        self,
        resnet: str = "resnet50",
        img_embedding_dim: int = 2048,
        text_embedding_dim: int = 768,
        temperature: float = 1.0,
    ):
        super(DualEncoder, self).__init__()
        self.image_encoder = ImageEncoder(resnet)
        self.text_encoder = TextEncoder()
        self.image_mapper = ProjectionHead(img_embedding_dim)
        self.text_mapper = ProjectionHead(text_embedding_dim)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_mapper(image_features)
        text_embeddings = self.text_mapper(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = nn.functional.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = loss_fn(logits, targets, reduction="none")
        images_loss = loss_fn(logits.T, targets.T, reduction="none")
        loss = (images_loss + texts_loss) / 2.0
        return loss.mean()
    

def loss_fn(outputs, targets):
  return nn.CrossEntropyLoss(reduction="none")(outputs, targets)
    

class TextEncoder(nn.Module):
    def __init__(self, model_name: str = TOKENIZER_NAME):
        super(TextEncoder, self).__init__()
        self.model = DistilBertModel.from_pretrained(model_name)
            
        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output[0]
        embedding = hidden_state[:, 0]
        return embedding
    

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(self, model_name: str = "resnet50"):
        super(ImageEncoder, self).__init__()
        if model_name == "resnet18":
            resnet = models.resnet18(pretrained=True)
        elif model_name == "resnet50":
            resnet = models.resnet50(pretrained=True)
        else:
            raise ValueError("Incorrect type of ResNet architecture.")
        
        self.model = torch.nn.Sequential(*(list(resnet.children())[:-1]))

        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, images):
        return self.model(images).squeeze()
    

class ProjectionHead(nn.Module):
    def __init__(
        self, embedding_dim: int, project_dim: int = PROJECTION_DIM, dropout: float = 0.2
    ) -> None:
        super(ProjectionHead, self).__init__()
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