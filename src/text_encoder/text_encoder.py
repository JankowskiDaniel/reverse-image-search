import torch
from torch import nn

from transformers import DistilBertModel

class TextEncoder(nn.Module):
    def __init__(self) -> None:
        super(TextEncoder, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output[0]
        embedding = hidden_state[:, 0]
        return embedding