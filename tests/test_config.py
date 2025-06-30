from opendrive_xai import Config, get_logger


def test_config_defaults():
    cfg = Config()
    assert cfg.batch_size == 4
    assert cfg.data_dir.exists()


def test_logger():
    logger = get_logger("test")
    logger.info("logger works") 