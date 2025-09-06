import torch
from .swin_transformer_modules import SwinTransformer

def my_swin_tiny_patch4_window7_224(num_classes: int = 1, **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SwinTransformer(
        in_chans=3,
        patch_size=4,
        window_size=7,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        num_classes=num_classes,
        **kwargs
    )
    return model.to(device), device
