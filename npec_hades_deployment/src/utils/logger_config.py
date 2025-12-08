import logging

# Create and configure the logger
logger = logging.getLogger('my_logger')
logger.setLevel(logging.ERROR)

# Create a console handler and set its level to ERROR
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)

# Ensure the root logger is set to ERROR
logging.getLogger().setLevel(logging.ERROR)
