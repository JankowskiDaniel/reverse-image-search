from src.image_encoder.image_encoder import ImageEncoder
from src.text_encoder.text_encoder import TextEncoder
from torch import nn

from projection_model import ProjectionHead


class DualEncoder(nn.Module):
    def __init__(
        self,
        resnet: str = "resnet18",
        img_embedding_dim: int = 512,
        text_embedding_dim: int = 768,
        temperature: float = 1.0,
    ):
        super(DualEncoder, self).__init__()
        self.resnet_encoder = ImageEncoder(resnet)
        self.bert_encoder = TextEncoder()
        self.image_mapper = ProjectionHead(img_embedding_dim)
        self.text_mapper = ProjectionHead(text_embedding_dim)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.resnet_encoder(batch["image"])
        text_features = self.bert_encoder(
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
        texts_loss = loss_fn(logits, targets)
        images_loss = loss_fn(logits.T, targets.T)
        loss = (images_loss + texts_loss) / 2.0
        return loss.mean()


def loss_fn(outputs, targets):
  return nn.CrossEntropyLoss(reduction="none")(outputs, targets)