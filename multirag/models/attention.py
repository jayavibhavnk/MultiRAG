import torch
import torch.nn as nn

class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        
    def forward(self, text_embeddings: torch.Tensor, image_embeddings: torch.Tensor):
        return self.attention(
            text_embeddings, 
            image_embeddings, 
            image_embeddings
        )[0]
