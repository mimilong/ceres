import codecs
import os
from os.path import dirname
from importlib import import_module

from sklearn.tree import _tree
from sklearn import tree
import json
import _pickle as cPickle
from ..MdBase import *
from ..utils import *
from .. import statfun as sf

# import_module(".MdVarSelect.{}.{}".format("MdVarSelectIV", "MdVarSelectIV"), __name__)

mapping = {}
for p in os.listdir(os.path.abspath(dirname(__file__))):
    p = p.split(".")[0]
    if p !=  "MdVarSelectApi" and p.startswith("MdVarSelect"):
        # print(__name__, p)
        mapping[p.split("MdVarSelect")[-1]] = getattr(import_module("..{}".format(p), __name__), p)

# mapping = {"IV":"MdVarSelect", "Xgb":"MdVarSelectXgb"}
# 变量选择
class MdVarSelectApi(MdBase):
    def __init__(self, wkdir=".", target_type='b', selector = {"IV":{"threshold": 0.01}}, logger=logging.getLogger('root'), indents = 1, log_stack = [], *args, **kw):
        MdBase.__init__(self, logger, indents, log_stack)
        self.wkdir = wkdir
        self.target_type = target_type
        self.method = selector
        self.kw = kw
        self.info("ModelTool MdVarSelect: Initial Success")

    @md_std_log()
    def fit(self, X, y, save = "all", *args):

        if type(X) is pd.DataFrame:
            X = {k: np.array(X[k]) for k in X.columns}

        stat_varperf = pd.DataFrame({"variable": list(X.keys())})

        # print(stat_varperf)

        for k, v in self.method.items():
            params = dict(v, target_type = self.target_type, log_stack = self.log_stack, indents = self.indents)
            params = dict(params, **self.kw)
            m = mapping[k](**params)
            m.fit(X = X,y = y)
            # print(m.varsele_get_perf())
            stat_varperf = pd.merge(stat_varperf, m.varsele_get_perf(), how = "left", on = "variable")

        self.stat_varperf = stat_varperf
        self.model_varsele_vars = self.varsele_vote()

        self.save(save = save)


    @md_std_log()
    def varsele_vote(self, vote = 1, **kw):
        if kw == {}:
            kw = {k:{"threshold": v.get("threshold"), "top": v.get("top")} for k,v in self.method.items() }

        score = 0
        for k, v in kw.items():
            k = k.lower()
            if v.get("threshold") is not None:
                tmp = (np.array(self.stat_varperf[k]) > v.get("threshold"))
                score = score + np.where(np.isnan(tmp), 0, tmp)
            else:
                arr = np.array(-self.stat_varperf[k])
                tmp = (arr.argsort().argsort() < v.get("top"))
                score = score + np.where(np.isnan(arr), 0, tmp)

        return list(self.stat_varperf["variable"][score >= vote])


    @md_std_log()
    def predict(self, X, subset = None,*args, **kw):
        vars = subset if subset is not None else self.model_varsele_vars
        return {k: X[k] for k in vars}

    @md_std_log()
    def varsele_get_perf(self):
        return self.stat_varperf

    def view(self):
        return self.model_varsele_vars, None

    @md_std_log()
    def save(self, save="all", *args, **kw):
        if save in ["obj", "all"]:
            file_path = os.path.join(self.wkdir, "obj", "obj_varsele.pkl")
            with open(file_path, 'wb') as f:
                cPickle.dump(self, f)

        if save in ["stat", "all"]:
            path = os.path.join(self.wkdir, "stat")
            self.stat_varperf.to_csv(path + "/stat_varperf.csv", index=False)

    @md_std_log()
    def export(self, lang="pmml", *args, **kw):
        path = os.path.join(self.wkdir, "model")
        if lang == "pmml":
            pass

        elif lang == 'json':
            with codecs.open('{}/{}'.format(path, 'model_varsele_vars.json'), 'w', 'utf-8') as f:
                json.dump(self.model_varsele_vars, f, ensure_ascii=False)

    def load(self, path = None):
        path = "model" if path is None else path
        with codecs.open('{}/{}'.format(path, 'model_varsele_vars.json')) as f:
            self.model_varsele_vars = json.load(f)
