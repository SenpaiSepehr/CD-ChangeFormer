
"""
reference: github.com/lucidrains/mlp-mixer-pytorch
"""

from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

pair = lambda x: x if isinstance(x, tuple) else (x, x)
def MLPMixer(*, image_size, channels, patch_size, dim, depth, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size) #4
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    input_dim = (patch_size ** 2) * channels
    h = image_h // patch_size
    w = image_w // patch_size
    
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size), #(8,512,2*4,2*4) -> (8,2*2,4*4*512)
        nn.Linear(input_dim, dim), # input -> 512
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last)),   #(512,FeedFWD(512,0.5,0.,nn.Linear)), (8,4,512)
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)) #(512,FeedFWD(4,4,0,conv1d))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        nn.Linear(dim,input_dim), # 512 -> input
        Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = h, w = w,
                  p2 = patch_size, p1 = patch_size)
    )