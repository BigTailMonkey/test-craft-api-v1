import logging as dev_logging
import os

from dotenv import load_dotenv

load_dotenv()

class Config:


    PROJECT_ID = os.environ.get("PROJECT_ID")
    ENVIRONMENT = os.environ.get("FLASK_ENV", "production")
    API_KEY = os.environ.get("OPENAI_API_KEY")
    LOG_NAME = "openai-api-proxy-log"
    AI_SERVER_URL = os.environ.get("AI_SERVER_URL","https://api.deepseek.com")  # 默认API地址

    LOG_LEVEL = dev_logging.DEBUG
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILENAME = 'app.log'
    logger = dev_logging.getLogger(LOG_NAME)
    logger.setLevel(LOG_LEVEL)
    handler = dev_logging.StreamHandler()
    handler.setFormatter(dev_logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)
