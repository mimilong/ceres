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
    def __init__(self, wkdir=".", target_type='b', var_desc=None, flag_set = ["FMS", "FFP", "FCP"], mapper = {"woe":{}}, logger=logging.getLogger('root'), indents = 1, log_stack = [], *args, **kw):
        MdBase.__init__(self, logger, indents, log_stack)
        self.wkdir = wkdir
        self.target_type = target_type
        self.method = mapper
        self.var_desc = var_desc
        self.flag_set = flag_set
        self.kw = kw

        self.info("ModelTool MdVarTransform: Initial Success")

    @md_std_log()
    def fit(self, X, y, save = "all", *args):
        if type(X) is pd.DataFrame:
            X = {k: np.array(X[k]) for k in X.columns}

        self.get_var_type(X = X)
        self.get_var_binary()

        transformer = []

        for k, v in self.method.items():
            params = dict(v, target_type=self.target_type, log_stack=self.log_stack, indents=self.indents, flag_set = self.flag_set, var_desc = self.var_desc)
            params = dict(params, **self.kw)
            m = mapping[k](**params)
            m.fit(X=X, y=y, save=None)
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

    @md_std_log()
    def get_var_type(self, X=None):
        if self.var_desc is None:
            self.var_desc = {k: {"origin": k, "desc": k, "type": "numeric" if sf.is_numeric(v) else "nominal"} for k, v
                             in X.items()}
        self.load_var_type()
        self.logger.debug(self.vars_num)

    @md_std_log()
    def get_var_binary(self):
        res = set()
        for f in self.flag_set:
            res = res | {k for k in self.vars_num if k.startswith("{}_".format(f))}
            res = res | {k for k in self.vars_num if k.startswith("{}".format(f)) and k[:5][-1] == "_"}
        self.vars_bin = res

    def view(self):
        return None, self.var_desc

    @md_std_log()
    def predict(self, X, subset = None, **kw):
        if type(X) is pd.DataFrame:
            X = {k: np.array(X[k]) for k in X.columns}
        res = {}
        subset =set(X.keys()) - self.vars_bin if subset is None else subset - self.vars_bin

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


        path = os.path.join(self.wkdir, "model")
        if lang == "pmml":
            pass

        elif lang == 'json':
            with codecs.open('{}/{}'.format(path, 'model_vartrans.json'), 'w', 'utf-8') as f:
                f.write(json_tricks.dumps(self.vars_bin, primitives=True))

    @md_std_log()
    def load(self, path = "model", lang="json"):
        transformer = []

        for k, v in self.method.items():
            params = dict(v, target_type=self.target_type, log_stack=self.log_stack, indents=self.indents,
                          flag_set=self.flag_set)
            params = dict(params, **self.kw)
            m = mapping[k](**params)
            m.load(path)
            transformer.append(m)
        self.transformer = transformer

        if lang == "json":
            with codecs.open('{}/{}'.format(path, 'model_vartrans.json')) as f:
                self.vars_bin = set(json.load(f))


    def load_var_type(self):
        self.vars_num = {k for k, v in self.var_desc.items() if v["type"] == "numeric"}
        self.vars_cat = {k for k, v in self.var_desc.items() if v["type"] in ["nominal", "ordinal"]}
