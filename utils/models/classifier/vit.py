# from tkinter import image_names
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from utils.register.registers import MODELS

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

@MODELS.register
class ViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self._cfg = cfg
        self._init_from_cfg()

        image_height, image_width = pair(self._image_size)
        patch_height, patch_width = pair(self._patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = self._channels * patch_height * patch_width
        assert self._pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, self._dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, self._dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self._dim))
        self.dropout = nn.Dropout(self._emb_dropout)

        self.transformer = Transformer(self._dim, self._depth, self._heads, self._dim_head, self._mlp_dim, self._dropout)

        self.pool = self._pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self._dim),
            nn.Linear(self._dim, self._num_classes)
        )

    def _init_from_cfg(self):
        self._image_size = self._cfg.image_size
        self._patch_size = self._cfg.patch_size
        self._num_classes = self._cfg.num_classes
        self._dim = self._cfg.dim
        self._depth = self._cfg.depth
        self._heads= self._cfg.heads
        self._mlp_dim = self._cfg.mlp_dim
        self._pool = self._cfg.pool
        self._channels = self._cfg.channels
        self._dim_head = self._cfg.dim_head
        self._dropout = self._cfg.dropout
        self._emb_dropout = self._cfg.emb_dropout
        self._patch_size = self._cfg.patch_size


    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
       
        x = self.to_latent(x)

        x = self.mlp_head(x)

        return x


if __name__ == "__main__":
    from utils.fileio.config import Config
    cfg = Config.fromfile("/home/shiqisun/train_framework/test_train_code/configs/CIFAR10/vit/cifar10_vit_0613_clean.py")
    model = ViT(cfg.model)
    x = torch.randn(1, 3, 32,32)
    y = model(x)
    print(y.size())