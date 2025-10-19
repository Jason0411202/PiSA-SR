import torch
import torch.nn as nn
import numpy as np

from basicsr.archs.arch_util import default_init_weights

class DegradationConditionEncoder(nn.Module):
    """
    Encode degradation condition (e.g., blur/noise level)
    into LoRA modulation matrices for VAE and UNet.
    """
    def __init__(
                self, 
                num_embeddings, 
                block_embedding_dim,
                num_unet_blocks, 
                lora_rank_unet,   
            ):
        super().__init__()

        self.W = nn.Parameter(torch.randn(num_embeddings), requires_grad=False)
        self.unet_de_mlp = nn.Sequential(
            nn.Linear(num_embeddings * 4, 256),
            nn.ReLU(True),
        )
        self.unet_block_mlp = nn.Sequential(
            nn.Linear(block_embedding_dim, 64),
            nn.ReLU(True),
        )
        self.unet_fuse_mlp = nn.Linear(256 + 64, lora_rank_unet ** 2)

        default_init_weights([self.unet_de_mlp, self.unet_block_mlp, self.unet_fuse_mlp], 1e-5)

        # embeddings for each block in UNet
        self.unet_block_embeddings = nn.Embedding(num_unet_blocks, block_embedding_dim)

    def forward(self, deg_score):
        """
        Args:
            deg_score: degradation score [B, num_embeddings * 4]
            vae_block_embeds: [N_vae_blocks, block_embedding_dim]
            unet_block_embeds: [N_unet_blocks, block_embedding_dim]
        Returns:
            vae_embeds: [B, N_vae_blocks, lora_rank_vae^2]
            unet_embeds: [B, N_unet_blocks, lora_rank_unet^2]
        """

        # 將 degradation condition 透過 Fourier encoding 展開
        deg_proj = deg_score[..., None] * self.W[None, None, :] * 2 * np.pi
        deg_proj = torch.cat([torch.sin(deg_proj), torch.cos(deg_proj)], dim=-1)   # Fourier 展開
        deg_proj = torch.cat([deg_proj[:, 0], deg_proj[:, 1]], dim=-1)             # 拼回單向量 [B, 2*W_dim]


        # Step 1. Project degradation condition
        unet_de_c_embed = self.unet_de_mlp(deg_proj)

        # Step 2. Project block embeddings
        unet_block_c_embeds = self.unet_block_mlp(self.unet_block_embeddings.weight)

        # Step 3. Fuse condition + block embedding → modulation vector
        unet_embeds = self.unet_fuse_mlp(torch.cat([
            unet_de_c_embed.unsqueeze(1).repeat(1, unet_block_c_embeds.size(0), 1),
            unet_block_c_embeds.unsqueeze(0).repeat(unet_de_c_embed.size(0), 1, 1)
        ], dim=-1))

        return unet_embeds
