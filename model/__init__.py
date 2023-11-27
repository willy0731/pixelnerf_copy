from .models import PixelNeRFNet


def make_model(conf, *args, **kwargs):
    """ Placeholder to allow more model types """
    # 若conf中沒有"type" 則設model_type = "pixelnerf"
    model_type = conf.get_string("type", "pixelnerf")  # pixelnerf
    if model_type == "pixelnerf":
        net = PixelNeRFNet(conf, *args, **kwargs)
    else:
        raise NotImplementedError("Unsupported model type", model_type)
    return net
