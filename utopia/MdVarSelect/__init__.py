import os
from os.path import dirname
from importlib import import_module

from .MdVarSelectApi import MdVarSelectApi

for p in os.listdir(os.path.abspath(dirname(__file__))):
    p = p.split(".")[0]
    if p != "MdVarSelectApi" and p.startswith("MdVarSelect"):
        # print(__name__, p)
        vars()[p] = getattr(import_module(".{}".format(p), __name__), p)