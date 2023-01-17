from src.image_encoder.image_encoder import ResNetEncoder
from src.text_encoder.text_encoder import TextEncoder

import pytest


def test_image_encoder_model_type():
    with pytest.raises(ValueError):
        ResNetEncoder("resnet101")
    
def test_image_encoder_features_number():
    resnet_encoder = ResNetEncoder()
    params = [len(parameter) for parameter in resnet_encoder.parameters()]
    assert params[-1] == 512

    resnet_encoder = ResNetEncoder("resnet50")
    params = [len(parameter) for parameter in resnet_encoder.parameters()]
    assert params[-1] == 2048

