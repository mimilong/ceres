import logging
import json
import time
import os
import numpy as np
import functools
import pandas as pd
from . import statfun as pf
import traceback


def set_logger(postfix = "", logerr = "error" , loginfo = "run", path = None, logname = None):

    logname = "root" if logname is None else logname
    postfix = "" if postfix == "" else "_" + postfix

    err_path = os.path.join(path if path is not None else "log" , logerr + postfix + ".log")
    info_path = os.path.join(path if path is not None else "log", loginfo + postfix + ".log")

    # %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s
    formatter = logging.Formatter('%(asctime)s\t%(filename)s[line:%(lineno)d]\t%(levelname)s:%(message)s')

    ferr = logging.FileHandler(err_path, mode="w")
    ferr.setLevel(logging.ERROR)
    ferr.setFormatter(formatter)
    logging.getLogger(logname).addHandler(ferr)

    finfo = logging.FileHandler(info_path, mode="w")
    finfo.setLevel(logging.INFO)
    finfo.setFormatter(formatter)
    logging.getLogger(logname).addHandler(finfo)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logging.getLogger(logname).addHandler(ch)

    logging.getLogger(logname).setLevel(logging.DEBUG)


def md_std_log(parallel=False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kw):
            if parallel is not True:
                self.indents = self.indents + 1
                self.log_stack.append("({})".format(func.__name__))
            # print(self.wkdir)

            try:
                start_time = time.time()

                self.logger.info("{}[{}] - {}".format(self.indents * "\t", "-".join(self.log_stack),
                                              "start"))
                params = {k: pf.obj_info(v) for k, v in kw.items()}
                # print(params)
                params = [[pf.obj_info(l) for l in args], params]

                self.logger.info("{}[{}] - {}".format(self.indents * "\t", "-".join(self.log_stack),
                                              json.dumps(params)))

                res = func(self, *args, **kw)

            except Exception as e:
                self.logger.error("{}[{}] - {}{}".format(self.indents * "\t", "-".join(self.log_stack),
                                                      "ERROR: ",traceback.format_exc()))
                raise
            finally:
                time_use = time.time() - start_time
                self.logger.info("{}[{}] - {}{}".format(self.indents * "\t", "-".join(self.log_stack),
                                                      "FINISH: " ,json.dumps({"time_use": time_use})))
                if parallel is not True:
                    self.indents = self.indents - 1
                    self.log_stack.pop()

            return res

        return wrapper

    return decorator





