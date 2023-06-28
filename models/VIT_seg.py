from functools import partial
import torch
import torch.nn as nn
from models.vision_transformer import VisionTransformer as VIT
from models.pos_embed import interpolate_pos_embed

def vit_base_patch16(**kwargs):
    model = VIT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VIT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VIT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# if __name__ == "__main__":
#     # VIT-Large 
#     model = vit_base_patch16(img_size=224, weight_init="nlhb",  num_classes=4, out_indices = [2, 5, 8, 11])
#     checkpoint_model = torch.load('../vit_base.pth')['model']
#     state_dict = model.state_dict()
#     for k in ['head.weight', 'head.bias']:
#         if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
#             print(f"Removing key {k} from pretrained checkpoint")
#             del checkpoint_model[k]

#     # interpolate position embedding
#     interpolate_pos_embed(model, checkpoint_model)

#     # load pre-trained model
#     model.load_state_dict(checkpoint_model, strict=False)
#     img = torch.randn(1, 3, 224, 224).cpu()
#     results, f = model(img)
#     print(results.shape)
#     print(f[0].shape,f[1].shape,f[2].shape,f[3].shape)
#     # preds = SETRNet(img)
#     # for output in preds:
#     #     print("output: ",output.size())
 
