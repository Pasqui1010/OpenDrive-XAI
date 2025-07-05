from opendrive_xai.config import OpenDriveXAIConfig
from opendrive_xai import get_logger


def test_config_defaults():
    cfg = OpenDriveXAIConfig()
    assert cfg.device == "auto"
    assert cfg.debug_mode == False


def test_logger():
    logger = get_logger("test")
    logger.info("logger works")
