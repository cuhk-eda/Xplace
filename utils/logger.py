import logging
import os
import sys


class CustomFormatter(logging.Formatter):
    white = "\x1b[37;21m"
    blue = "\x1b[34;21m"
    yellow = "\x1b[33;21m"
    green = "\x1b[32;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    # time_format = "[%(asctime)s.%(msecs)03d] "
    # FIXME: An elapsed time gap exists between the C++ Timer and Python Timer,
    #        but I don't know how to resolve it...
    time_format = "[%(relativeCreatedSecond)4d.%(relativeCreatedMSecond)03d] "
    debug_msg = " (%(module)s.py Line%(lineno)d) %(msg)s"
    FORMATS = {
        logging.DEBUG: time_format + blue + "DEBUG" + reset + debug_msg,
        logging.INFO: time_format + "%(msg)s",
        logging.WARNING: time_format + yellow + "WARNING" + reset + " %(msg)s",
        logging.ERROR: time_format + red + "ERROR" + reset + " %(msg)s",
        logging.CRITICAL: time_format + bold_red + "CRITICAL" + reset + " %(msg)s",
    }

    def __init__(self):
        super().__init__(datefmt="%H:%M:%S")

    def format(self, record):
        record.relativeCreatedSecond = record.relativeCreated / 1000
        record.relativeCreatedMSecond = record.relativeCreated % 1000
        self._style._fmt = self.FORMATS.get(record.levelno)
        return logging.Formatter.format(self, record)


def setup_logger(args, sys_argv) -> logging.Logger:
    res_root = os.path.join(args.result_dir, args.exp_id)
    log_file_path = os.path.join(res_root, args.log_dir, args.log_name)
    if not os.path.exists(os.path.dirname(log_file_path)):
        os.makedirs(os.path.dirname(log_file_path))

    formatter = CustomFormatter()
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(screen_handler)

    logger.info("Command line: python " + " ".join(sys_argv))
    logger.info("log file at {}".format(log_file_path))
    logger.info("")
    for arg in vars(args):
        logger.info("{}: {}".format(arg, getattr(args, arg)))
    logger.info("")

    logging.getLogger("matplotlib.font_manager").disabled = True
    return logger
