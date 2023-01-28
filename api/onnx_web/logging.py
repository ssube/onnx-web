from logging import getLogger
from logging.config import dictConfig
from os import path
from yaml import safe_load

logging_path = './logging.yaml'

# setup logging config before anything else loads
try:
  if path.exists(logging_path):
    with open(logging_path, 'r') as f:
        config_logging = safe_load(f)
        # print('configuring logger', config_logging)
        dictConfig(config_logging)
except Exception as err:
  print('error loading logging config: %s' % (err))
