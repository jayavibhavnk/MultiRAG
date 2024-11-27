import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import numpy as np

class MultiModalEncoder:
    def __init__(
        self,
        text_model_name: str = 'sentence-transformers/all-mpnet-base-v2',
        clip_model_name: str = "openai/clip-vit-base-patch32"
    ):
        self.text_model = SentenceTransformer(text_model_name)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        self.text_dimension = self.text_model.get_sentence_embedding_dimension()
        self.image_dimension = self.clip_model.config.projection_dim
        
    def encode_text(self, text: str) -> np.ndarray:
        return self.text_model.encode(text, convert_to_numpy=True)
        
    def encode_image(self, image) -> np.ndarray:
        inputs = self.clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        return image_features.numpy()
