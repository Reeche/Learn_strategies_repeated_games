import logging

# Gets or creates a logger
root_logger = logging.getLogger()

# set log level
root_logger.setLevel(logging.INFO)

# define format
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')

# define file handler and set formatter
file_handler = logging.FileHandler('results/logfile_13.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

# add file handler to logger
root_logger.addHandler(file_handler)
root_logger.addHandler(stream_handler)
