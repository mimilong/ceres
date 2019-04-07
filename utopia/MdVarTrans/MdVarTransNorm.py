import codecs
import json_tricks

from ..MdBase import *
from ..MdProcess import MdProcess
import _pickle as cPickle
from ..utils import *
from .. import statfun as sf

class MdVarTransNorm(MdBase):
    def __init__(self, wkdir=".",
                 logger=logging.getLogger('root'), indents=1, log_stack=[], *args, **kw):
        MdBase.__init__(self, logger, indents, log_stack)
        self.wkdir = wkdir
        # self.flag_set = flag_set
        self.kw = kw

        self.info("ModelTool MdVarTransNorm: Initial Success")

    @md_std_log()
    def fit(self, X, save="all", *args, **kw):
        if type(X) is not pd.DataFrame:
            X = pd.DataFrame(X)

        obj_processor = MdProcess(wkdir=self.wkdir, logger=self.logger, indents=self.indents, log_stack=self.log_stack, **self.kw)
        obj_processor.fit(X=X, y=1, save=None)

        self.stat_vartans_norm = obj_processor.stat_eda_num
        self.model_vartans_norm = obj_processor.model_proc_num

        self.save(save=save)


    def view(self):
        pass

    @md_std_log()
    def predict(self, X, subset=None, m = "norm", **kw):
        if type(X) is pd.DataFrame:
            X = {k: np.array(X[k]) for k in X.columns}
        vars_num = list(X.keys()) if subset is None else list(set(X.keys()) & set(subset))

        if m == "norm":
            return {k:(X[k] - self.model_vartans_norm[k]["mean"])/self.model_vartans_norm[k]["std"] for k in vars_num}

        if m == "minmax":
            return {k: (X[k] - self.model_vartans_norm[k]["min"]) / (self.model_vartans_norm[k]["max"] - self.model_vartans_norm[k]["min"]) for k in
                    vars_num}

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
            file_path = os.path.join(self.wkdir, "obj", "obj_vartrans_norm.pkl")
            with open(file_path, 'wb') as f:
                cPickle.dump(self, f)

        if save in ["stat", "all"]:
            path = os.path.join(self.wkdir, "stat")
            self.stat_vartans_norm.to_csv(path + "/stat_vartans_norm.csv", index=False)

    @md_std_log()
    def export(self, lang="pmml", subset=None, *args, **kw):
        path = os.path.join(self.wkdir, "model")
        if lang == "pmml":
            pass

        elif lang == 'json':
            with codecs.open('{}/{}'.format(path, 'model_vartans_norm.json'), 'w', 'utf-8') as f:
                json.dump(self.model_vartans_norm, f, ensure_ascii=False)

    def load(self, path = "model", lang="json"):
        if lang == "json":
            with codecs.open('{}/{}'.format(path, 'model_vartans_norm.json')) as f:
                self.model_vartans_norm = json.load(f)
