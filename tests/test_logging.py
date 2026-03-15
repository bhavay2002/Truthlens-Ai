import logging


def test_logger_initialization():

    logger = logging.getLogger("truthlens")

    logger.info("test log")

    assert logger is not None