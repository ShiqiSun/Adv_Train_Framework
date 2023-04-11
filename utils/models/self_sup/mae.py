
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block

from utils.register.registers import MODELS



@MODELS.register
class MaskedAutoencoderViT(nn.Module):
    def __init__(self, 
                img_size: int = 224, 
                patch_size: int = 16, 
                in_chans:int = 3,
                embed_dim:int = 1024, 
                depth:int = 24, 
                num_heads:int = 16,
                decoder_embed_dim:int = 512, 
                decoder_depth:int = 8, 
                decoder_num_heads:int = 16,
                mlp_ratio:int = 4., 
                norm_layer:int = nn.LayerNorm, 
                norm_pix_loss:int = False) -> None:
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), 
                                        requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), 
                                                requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, 
                    mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        """
        initialize models weights
        initialize (and freeze) pos_embed by sin-cos embedding
        """
        pos_embed = 1

        pass

    def forward(self, imgs):
        imgs = self.patch_embed(imgs)
        print(imgs.shape)
        for blk in self.blocks:
            imgs = blk(imgs)
            print(imgs.shape)
        return imgs

if __name__ == "__main__":
    model = MaskedAutoencoderViT()
    imgs = torch.rand(1, 3, 224, 224)
    x = model(imgs)
    print