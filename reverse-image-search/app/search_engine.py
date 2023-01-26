import torch
from architecture import DualEncoder
import pandas as pd
from typing import List
from transformers import DistilBertTokenizer
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

from settings import (MODEL_PATH,
                      IMG_EMB_PATH,
                      TEXTS_EMB_PATH,
                      CAPTIONS_PATH,
                      TOKENIZER_NAME,
                      DEVICE)

class SearchEngine:
    def __init__(self,
                 model_path: str = MODEL_PATH,
                 img_embeddings_path: str = IMG_EMB_PATH,
                 texts_embeddings_path: str = TEXTS_EMB_PATH,
                 captions_path: str = CAPTIONS_PATH):
        self.__init__model(model_path)
        self.img_embeddings = torch.load(img_embeddings_path, map_location=torch.device('cpu'))
        self.text_embeddings = torch.load(texts_embeddings_path, map_location=torch.device('cpu'))
        self.captions_df = pd.read_csv(captions_path)
        self.tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_NAME)
        self.transform = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Resize((224, 224)),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        ]
                    )


    def __init__model(self, model_path: str) -> None:
        self.model = DualEncoder().to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

    def match_query_to_image(self, query: str, max_results: int = 9) -> List[int]:
        encoded_query = self.tokenizer([query])

        batch = {
            key: torch.tensor(values).to(DEVICE)
            for key, values in encoded_query.items()
        }

        with torch.no_grad():
            text_features = self.model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = self.model.text_mapper(text_features)
    
        image_embeddings_n = F.normalize(self.img_embeddings, p=2, dim=-1)
        text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
        dot_similarity = text_embeddings_n @ image_embeddings_n.T
        
        scores, indices = torch.topk(dot_similarity.squeeze(0), max_results)
        return scores.detach().numpy().tolist(), indices.detach().numpy().tolist()
    
    def match_image_to_texts(self, image: np.array, max_results: int = 5) -> List[str]:
        image = self.transform(image)
        image = image.unsqueeze(0)

        with torch.no_grad():
            image_features = self.model.image_encoder(image.to(DEVICE))
            image_embeddings = self.model.image_mapper(image_features)

        image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings_n = F.normalize(self.text_embeddings, p=2, dim=-1)
        dot_similarity = image_embeddings_n @ text_embeddings_n.T
        
        scores, indices = torch.topk(dot_similarity.squeeze(0), max_results)
        return scores.detach().numpy().tolist(), indices.detach().numpy().tolist()
    
    def find_similar_images(self, image: np.array, max_results: int = 9) -> List[int]:
        image = self.transform(image)
        image = image.unsqueeze(0)

        with torch.no_grad():
            image_features = self.model.image_encoder(image.to(DEVICE))
            single_embedding = self.model.image_mapper(image_features)
  
        image_embeddings_n = F.normalize(self.img_embeddings, p=2, dim=-1)
        single_embedding_n = F.normalize(single_embedding, p=2, dim=-1)
        dot_similarity = single_embedding_n @ image_embeddings_n.T

        scores, indices = torch.topk(dot_similarity.squeeze(0), max_results)
        return scores.detach().numpy().tolist(), indices.detach().numpy().tolist()