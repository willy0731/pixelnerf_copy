from .encoder import SpatialEncoder
from .resnetfc import ResnetFC

# MLP可使用ImplicitNet 或 ResnetFC
# 預設用resnet
def make_mlp(conf, d_in, d_latent=0, allow_empty=False, **kwargs):
    mlp_type = conf.get_string("type", "mlp")  # mlp | resnet
    if mlp_type == "resnet":
        net = ResnetFC.from_conf(conf, d_in, d_latent=d_latent, **kwargs)
    elif mlp_type == "empty" and allow_empty:
        net = None
    else:
        raise NotImplementedError("Unsupported MLP type")
    return net

# 選擇使用Spatial Encoder (PixelNeRF論文中提及)
def make_encoder(conf, **kwargs):
    enc_type = conf.get_string("type", "spatial")  # spatial | global
    if enc_type == "spatial":
        net = SpatialEncoder.from_conf(conf, **kwargs)
    else:
        raise NotImplementedError("Unsupported encoder type")
    return net
