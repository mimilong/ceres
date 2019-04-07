import codecs
import json_tricks
from os.path import dirname
from importlib import import_module

from ..MdBase import *
from ..MdProcess import MdProcess
import _pickle as cPickle
from ..utils import *
from .. import statfun as sf

mapping = {}
for p in os.listdir(os.path.abspath(dirname(__file__))):
    p = p.split(".")[0]
    if p !=  "MdVarTransApi" and p.startswith("MdVarTrans"):
        # print(__name__, p)
        mapping[p.split("MdVarTrans")[-1].lower()] = getattr(import_module("..{}".format(p), __name__), p)


# 变量变换的入参是字典数据数据框，出参为字典
class MdVarTransApi(MdBase):
    def __init__(self, wkdir=".", target_type='b', var_type=None, var_desc=None, method = {"woe":{"threshold": 0.01}}, logger=logging.getLogger('root'), indents = 1, log_stack = [], *args, **kw):
        MdBase.__init__(self, logger, indents, log_stack)
        self.wkdir = wkdir
        self.target_type = target_type
        self.method = method
        self.var_type = var_type
        self.var_desc = var_desc
        self.kw = kw

        self.info("ModelTool MdVarTransform: Initial Success")

    @md_std_log()
    def fit(self, X, y, save = "all", *args):
        if type(X) is pd.DataFrame:
            X = {k: np.array(X[k]) for k in X.columns}

        transformer = []

        for k, v in self.method.items():
            params = dict(v, target_type=self.target_type, log_stack=self.log_stack, indents=self.indents, var_type = self.var_type)
            params = dict(params, **self.kw)
            m = mapping[k](**params)
            m.fit(X=X, y=y)
            transformer.append(m)

        self.transformer = transformer
        self.update_varinfo()

        self.save(save=save)

    @md_std_log()
    def update_varinfo(self):
        tmp = self.var_desc
        for m in self.transformer:
            _, var_desc_t = m.view()
            tmp = dict(tmp, **var_desc_t)

        self.var_desc = tmp


    def view(self):
        return None, self.var_desc

    @md_std_log()
    def predict(self, X, subset = None, **kw):
        if type(X) is pd.DataFrame:
            X = {k: np.array(X[k]) for k in X.columns}
        res = {}
        for m in self.transformer:
            res = dict(res, **m.predict(X = X, subset = subset))
        return dict(res, **X)

    @md_std_log()
    def save(self, save="all", *args, **kw):
        """
        保存各个对象的变换数据
        :param save:
        :param args:
        :param kw:
        :return:
        """
        for m in self.transformer:
            m.save(save = save)

    @md_std_log()
    def export(self, lang="pmml", subset=None, *args, **kw):
        for m in self.transformer:
            m.export(lang = lang, subset = subset, *args, **kw)

    def load(self, path = "model"):
        for m in self.method:
            m.load(path)
