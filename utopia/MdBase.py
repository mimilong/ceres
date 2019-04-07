import os
import time
import logging
import functools

import traceback

# Ll class MdProcess(MdBase)
class MdBase(object):
    def __init__(self, logger=logging.getLogger('root'), indents = 1, log_stack = []):
        self.logger = logger
        self.indents = indents
        self.log_stack = log_stack + ["[{}]".format(self.__class__.__name__)]

    def info(self, message):
        self.logger.info("{}[{}] - {}".format(self.indents * "\t", "-".join(self.log_stack),
                                              message))

    def error(self):
        self.logger.error("{}[{}] - {}{}".format(self.indents * "\t", "-".join(self.log_stack),
                                                 "ERROR: ", traceback.format_exc()))


