import codecs
import json_tricks

from ..MdBase import *
import _pickle as cPickle
from ..utils import *
from .. import statfun as sf


class MdVarTransPower(MdBase):
    def __init__(self, wkdir=".", var_desc=None, flag_set = ["FMS", "FFP", "FCP"],
                 logger=logging.getLogger('root'), indents=1, log_stack=[], *args, **kw):
        MdBase.__init__(self, logger, indents, log_stack)
        self.wkdir = wkdir
        self.flag_set = flag_set
        self.var_desc = var_desc
        self.map = {"SQ":lambda x:x**2, "SR":np.sqrt, "IV":lambda x: 1/(1+x), "LN":lambda x:np.log(1+x)}

        self.info("ModelTool MdVarTransform: Initial Success")

    @md_std_log()
    def fit(self, save="all", *args, **kw):

        self.vars_num = {k for k, v in self.var_desc.items() if v["type"] == "numeric"}
        self.update_varinfo()
        self.save(save=save)

    @md_std_log()
    def update_varinfo(self, power_set = ["SQ", "SR", "IV", "LN"]):
        res = {}
        for m in power_set:
            var_desc_tmp = {"{}_{}".format(m,k):{"origin":k, "desc":"{}({})".format(m, self.var_desc[k]["desc"]), "type": "numeric"} for k in self.vars_num}
            res = dict(res, **var_desc_tmp)

        self.var_desc = res

    def view(self):
        return None, self.var_desc

    @md_std_log()
    def predict(self, X, subset=None, power_set = ["SQ", "SR", "IV", "LN"], **kw):
        if type(X) is pd.DataFrame:
            X = {k: np.array(X[k]) for k in X.columns}
        vars_num = self.vars_num if subset is None else list(set(self.vars_num) & set(subset))

        res = {}
        for m in power_set:
            res = dict(res, **{"{}_{}".format(m, k): self.map[m](X[k]) for k in vars_num})
        return dict(res)

    @md_std_log()
    def save(self, save="all", *args, **kw):
        """
        保存各个对象的变换数据
        :param save:
        :param args:
        :param kw:
        :return:
        """
        if save in ["obj", "all"]:
            file_path = os.path.join(self.wkdir, "obj", "obj_vartrans_power.pkl")
            with open(file_path, 'wb') as f:
                cPickle.dump(self, f)

        if save in ["stat", "all"]:
            path = os.path.join(self.wkdir, "stat")
            stat_vartans_power = pd.DataFrame([dict(v, variable=k) for k, v in self.var_desc.items()])
            stat_vartans_power.to_csv(path + "/stat_vartans_power.csv", index=False)

    @md_std_log()
    def export(self, lang="pmml", subset=None, *args, **kw):
        path = os.path.join(self.wkdir, "model")
        if lang == "pmml":
            pass

        elif lang == 'json':
            with codecs.open('{}/{}'.format(path, 'model_vartrans_power.json'), 'w', 'utf-8') as f:
                json.dump(self.var_desc, f, ensure_ascii=False)

    @md_std_log()
    def load(self, path = "model", lang="json", power_set = ["SQ", "SR", "IV", "LN"]):
        if lang == "json":
            with codecs.open('{}/{}'.format(path, 'model_vartrans_power.json')) as f:
                self.var_desc = json.load(f)
            self.vars_num = {v["origin"] for k,v in self.var_desc.items() if v["type"] == "numeric"}

