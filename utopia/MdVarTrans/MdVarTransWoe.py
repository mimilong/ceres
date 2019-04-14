import codecs
import json_tricks

from ..MdBase import *

from ..utils import *
from .. import statfun as sf
from .. import MdPublicClass as mp


# 使用WOE变换
class MdVarTransWoe(MdBase):

    def __init__(self, wkdir=".", miss_set = [np.nan], var_desc=None, flag_set = ["FMS", "FFP", "FCP"], logger=logging.getLogger('root'), indents=1, log_stack=[], *args, **kw):
        MdBase.__init__(self, logger, indents, log_stack)
        self.wkdir = wkdir
        self.miss_set = miss_set
        self.var_desc = var_desc
        self.flag_set = flag_set
        if kw.get("target_type"):
            kw.pop("target_type")
        self.binner = mp.MdUtilBinning(target_type='b', logger = logger, indents = indents, log_stack = log_stack, *args, **kw)

        self.info("ModelTool MdVarTransWoe: Initial Success")

    @md_std_log()
    def fit(self, X, y, save = "all", *args, **kw):
        if type(X) is pd.DataFrame:
            X = {k: np.array(X[k]) for k in X.columns}
        self.get_var_type(X = X)
        self.get_var_binary()
        stat_vartrans_woe = self.varstrans_num_fit(X = X, y = y)
        stat_vartrans_woe = pd.concat([stat_vartrans_woe, self.varstrans_cat_fit(X = X, y = y)])
        self.stat_vartrans_woe = stat_vartrans_woe

        self.update_varinfo()
        self.save(save=save)

    @md_std_log()
    def varstrans_num_fit(self, X, y):
        vars_num = set(self.vars_num) & set(X.keys())
        cuts = {k: self.binner.fit(X[k], y) for k in vars_num}
        woe_tbl = {k:sf.woe_tbl(sf.cut2(X[k], v, miss_set=self.miss_set), y).assign(variable = k) for k,v in cuts.items()}

        self.model_vartrans_woe_num = {}
        for k,v in cuts.items():
            woe = np.zeros(max(woe_tbl[k]["decile"] + 1))
            # self.info(k)
            # self.info(woe)
            # self.info(woe_tbl[k]["decile"])
            woe[woe_tbl[k]["decile"]] = woe_tbl[k]["woe"]
            self.model_vartrans_woe_num[k] = {"cut": v, "woe":woe}

        return pd.concat(list(woe_tbl.values()))

    @md_std_log()
    def varstrans_cat_fit(self, X, y):
        self.vars_cat = set(self.vars_cat) & set(X.keys())
        if len(self.vars_cat) == 0:
            self.model_vartrans_woe_cat = {}
            return pd.DataFrame()

        woe_tbl = {k:sf.woe_tbl(X[k], y).assign(variable = k) for k in self.vars_cat}
        self.model_vartrans_woe_cat = {k: v.set_index('decile')["woe"].to_dict() for k,v in woe_tbl.items()}

        # self.logger.debug(str(woe_tbl.values()))
        return pd.concat(list(woe_tbl.values()))

    @md_std_log()
    def update_varinfo(self):

        var_desc = {"WOE_{}".format(k):{"origin": k, "desc": "WOE for {}".format(self.var_desc[k]["desc"]), "type": "numeric"} for k in self.vars_num}
        var_desc = dict(var_desc, **{"WOE_{}".format(k):{"origin": k, "desc": "WOE for {}".format(self.var_desc[k]["desc"]), "type": "numeric"} for k in self.vars_cat})

        self.var_desc = var_desc
        self.model_vartrans_woe = {"numeric":self.model_vartrans_woe_num, "nominal":self.model_vartrans_woe_cat, "binary":self.vars_bin}

    def view(self):

        return self.stat_vartrans_woe, self.var_desc


    @md_std_log()
    def get_var_type(self, X = None):
        if self.var_desc is None:
            self.var_desc = {k:{"origin":k, "desc":k, "type":"numeric" if sf.is_numeric(v) else "nominal"} for k,v in X.items()}
        self.load_var_type()

    @md_std_log()
    def get_var_binary(self):
        res = set()
        for f in self.flag_set:
            res = res | {k for k in self.vars_num if k.startswith("{}_".format(f))}
            res = res | {k for k in self.vars_num if k.startswith("{}".format(f)) and k[:5][-1] == "_"}
        self.vars_bin = res

    # 返回具体分段或分类的woe值
    @md_std_log()
    def predict(self, X, subset = None):

        subset = set(X.keys()) if subset is None else subset
        vars_num = set(self.vars_num) & set(subset) - self.vars_bin
        vars_cat = set(self.vars_cat) & set(subset) - self.vars_bin
        vars_bin = set(self.vars_bin) & set(subset)

        result = {"WOE_{}".format(k): self.model_vartrans_woe["numeric"][k]["woe"][sf.cut2(X[k], self.model_vartrans_woe["numeric"][k]["cut"], self.miss_set)] for k in vars_num}
        # result = dict(result, **{k:v for k,v in X.items() if k in vars_bin})

        result_cat = {}
        for var in vars_cat:
            arr = 0
            woe_map = self.model_vartrans_woe["nominal"][var]
            for k,v in woe_map.items():
                arr = arr + (X[k] == k)*v
            result_cat["WOE_{}".format(var)] = arr

        return dict(result, **result_cat)

    # export to model folder,
    @md_std_log()
    def export(self, lang="pmml", *args, **kw):
        path = os.path.join(self.wkdir, "model")
        if lang == "pmml":
            pass

        elif lang == 'json':
            with codecs.open('{}/{}'.format(path, 'model_vartrans_woe.json'), 'w', 'utf-8') as f:
                f.write(json_tricks.dumps(self.model_vartrans_woe, primitives=True))
                # json.dump(self.model_vartrans_woe, f, ensure_ascii=False)

    @md_std_log()
    def load(self, path="model", lang="json"):
        if lang == "json":
            with codecs.open('{}/{}'.format(path, 'model_vartrans_woe.json')) as f:
                model_vartrans_woe = json.load(f)
                model_vartrans_woe["numeric"] = {k:{vk:np.array(vv) for vk, vv in v.items()} for k,v in model_vartrans_woe["numeric"].items()}
                model_vartrans_woe["nominal"] = {k: {vk: np.array(vv) for vk, vv in v.items()} for k, v in
                                                 model_vartrans_woe["nominal"].items()}
                self.model_vartrans_woe = model_vartrans_woe

        var_desc = {k:{"origin":k, "desc":k, "type":"numeric"} for k,v in self.model_vartrans_woe["numeric"].items()}
        self.var_desc = dict(var_desc, **{k:{"origin":k, "desc":k, "type":"nominal"} for k,v in self.model_vartrans_woe["nominal"].items()})
        self.load_var_type()
        self.vars_bin = set(self.model_vartrans_woe["binary"])

    def load_var_type(self):
        self.vars_num = {k for k, v in self.var_desc.items() if v["type"] == "numeric"}
        self.vars_cat = {k for k, v in self.var_desc.items() if v["type"] in ["nominal", "ordinal"]}

    # save to obj
    @md_std_log()
    def save(self, save, *args, **kw):
        pass
