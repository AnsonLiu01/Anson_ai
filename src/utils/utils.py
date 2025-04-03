from typing import Any, Dict
import yaml

from loguru import logger


@staticmethod
def load_yaml(
    config_path: str
) -> Dict[Any, Any]:
    """
    Function to load yaml file
    """
    logger.info('Loading yaml file')
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
