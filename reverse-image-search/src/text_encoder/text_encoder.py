import torch
from torch import nn

from transformers import DistilBertModel

from ..settings import TOKENIZER_NAME

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