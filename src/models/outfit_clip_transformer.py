import torch
from torch import nn
from typing import List, Tuple, Union
from ..data.datatypes import FashionItem
from dataclasses import dataclass
from .modules.encoder import CLIPItemEncoder,CLIPTextEncoder,HuggingFaceTextEncoder
from .modules.text_encoder import BaseTextEncoder
from .outfit_transformer import OutfitTransformer, OutfitTransformerConfig
import numpy as np

@dataclass
class OutfitCLIPTransformerConfig(OutfitTransformerConfig):
    item_enc_clip_model_name = "models/fashion-clip"
            

class OutfitCLIPTransformer(OutfitTransformer):
    
    def __init__(
        self, 
        cfg: OutfitCLIPTransformerConfig = OutfitCLIPTransformerConfig()
    ):
        super().__init__(cfg)

    def _init_item_enc(self) -> CLIPItemEncoder:
        """Builds the outfit encoder using configuration parameters."""
        self.item_enc = CLIPItemEncoder(
            model_name=self.cfg.item_enc_clip_model_name,
            enc_norm_out=self.cfg.item_enc_norm_out,
            aggregation_method=self.cfg.aggregation_method
        )
    
    def precompute_clip_embedding1(self, item: List[FashionItem]) -> np.ndarray:
        """Precomputes the encoder(backbone) embeddings for a list of fashion items."""
        outfits = [[item_] for item_ in item]
        images, texts, mask = self._pad_and_mask_for_outfits(outfits)
        # enc_outs = self.item_enc(images, texts) # [B, 1, D]
        encoder = HuggingFaceTextEncoder().to(self.device)
        enc_outs =encoder(texts)
        # if not hasattr(self, 'proj_to_128'):
        #     self.proj_to_128 = nn.Linear(1024, 128).to(self.device)
        # embeddings = self.proj_to_128(enc_outs[:, 0, :])  # [B, 128]
        embeddings = enc_outs[:, 0, :] # [B, D]
        print(embeddings.shape)
        return embeddings.detach().cpu().numpy()
    def precompute_clip_embedding(self, item: List[FashionItem]) -> np.ndarray:
        """Precomputes the encoder(backbone) embeddings for a list of fashion items."""
        outfits = [[item_] for item_ in item]
        images, texts, mask = self._pad_and_mask_for_outfits(outfits)
        enc_outs = self.item_enc(images, texts) # [B, 1, D]
        embeddings = enc_outs[:, 0, :] # [B, D]
        return embeddings.detach().cpu().numpy()

    

            

class OutfitCLIPTextTransformer(OutfitTransformer):
    
    def __init__(
        self, 
        cfg: OutfitCLIPTransformerConfig = OutfitCLIPTransformerConfig()
    ):
        super().__init__(cfg)
     
    def _init_text_enc(self) -> CLIPTextEncoder:
        """Builds the outfit encoder using configuration parameters."""
        self.text_enc = CLIPTextEncoder(
            model_name=self.cfg.item_enc_clip_model_name,
            enc_norm_out=self.cfg.item_enc_norm_out,
            aggregation_method=self.cfg.aggregation_method
        )
    
    def precompute_text_embedding(self, item: List[FashionItem]) -> np.ndarray:
        """纯文本嵌入计算（新增方法）"""
        outfits = [[item_] for item_ in item] 
        
        texts, mask = self._pad_and_mask_for_text(outfits)
        enc_outs = self.text_enc(texts) # [B, 1, D]
        embeddings = enc_outs[:, 0, :] # [B, D]
        
        return embeddings.detach().cpu().numpy()
    
