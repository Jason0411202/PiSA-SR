import torch
import torch.nn as nn
import numpy as np

from basicsr.archs.arch_util import default_init_weights

def my_lora_fwd(self, x: torch.Tensor, *args: any, **kwargs: any) -> torch.Tensor:
    """
    自定義的 LoRA forward function, 目的是為了在 forward 的過程中參考 degradation condition modulation vector (de_mod)
    """
    # self._check_forward_args(x, *args, **kwargs)
    adapter_names = kwargs.pop("adapter_names", None)

    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif adapter_names is not None:
        result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x = x.to(lora_A.weight.dtype)

            _tmp = lora_A(dropout(x))
            if isinstance(lora_A, torch.nn.Conv2d):
                _tmp = torch.einsum('...khw,...kr->...rhw', _tmp, self.de_mod)
            elif isinstance(lora_A, torch.nn.Linear):
                _tmp = torch.einsum('...lk,...kr->...lr', _tmp, self.de_mod)
            else:
                raise NotImplementedError('only conv and linear are supported yet.')

            result = result + lora_B(_tmp) * scaling

        result = result.to(torch_result_dtype)

    return result

class  DegradationConditionEncoder(nn.Module):
    """
    此模組負責將 degradation condition (通常是 [blur 等級, noise 等級]) Encoder 成可以用於 LoRA 的向量
    """
    def __init__(
                self, 
                num_embeddings, 
                block_embedding_dim,
                num_unet_blocks, 
                lora_rank_unet,   
            ):
        super().__init__()

        self.W = nn.Parameter(torch.randn(num_embeddings)) # 與作者不同的是, self.W 也是會學習參數的一部分並會透過 state_dict() 存下來, 為了避免 train 跟 test 的此參數不一致
        self.unet_de_mlp = nn.Sequential(
            nn.Linear(num_embeddings * 4, 256),
            nn.ReLU(True),
        )
        self.unet_block_mlp = nn.Sequential(
            nn.Linear(block_embedding_dim, 64),
            nn.ReLU(True),
        )
        self.unet_fuse_mlp = nn.Linear(256 + 64, lora_rank_unet ** 2)
        # default_init_weights([self.unet_de_mlp, self.unet_block_mlp, self.unet_fuse_mlp], 1e-5) # 這行會導致 LoRA 的 W + ACB 退化成 W

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


        # 透過 MLP 投影 degradation condition
        unet_de_c_embed = self.unet_de_mlp(deg_proj)

        # 取得 U-Net block ID embedding
        unet_block_c_embeds = self.unet_block_mlp(self.unet_block_embeddings.weight)

        # 結合 degradation condition 與 block embedding，產生 LoRA embedding
        unet_embeds = self.unet_fuse_mlp(torch.cat([
            unet_de_c_embed.unsqueeze(1).repeat(1, unet_block_c_embeds.size(0), 1),
            unet_block_c_embeds.unsqueeze(0).repeat(unet_de_c_embed.size(0), 1, 1)
        ], dim=-1))

        return unet_embeds
