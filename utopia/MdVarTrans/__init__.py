import os
from os.path import dirname
from importlib import import_module

from .MdVarTransApi import MdVarTransApi

for p in os.listdir(os.path.abspath(dirname(__file__))):
    p = p.split(".")[0]
    if p != "MdVarTransApi" and p.startswith("MdVarTrans"):
        # print(__name__, p)
        vars()[p] = getattr(import_module(".{}".format(p), __name__), p)