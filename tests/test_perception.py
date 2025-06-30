import torch

from opendrive_xai.perception import TinyBEVEncoder


def test_encoder_forward():
    model = TinyBEVEncoder(bev_channels=64)
    inp = torch.randn(2, 3, 224, 224)
    out = model(inp)
    assert out.shape == (2, 64, 7, 7) 