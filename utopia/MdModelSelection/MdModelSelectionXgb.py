import codecs

from scipy.special import expit
import _pickle as cPickle

import xgboost as xgb

from ..MdBase import *
from ..utils import *
from numpy.random import RandomState
from .. import statfun as sf


class MdModelSelectionXgb(MdBase):
    def __init__(self, wkdir=".", seed = 0, target_type = "b", logger=logging.getLogger('root'), indents=1, log_stack=[], *args, **kw):
        '''
        '''
        MdBase.__init__(self, logger, indents, log_stack)
        self.wkdir = wkdir
        self.target_type = target_type
        self.seed = seed
        self.kw = kw

        self.info("ModelTool MdModelSelectionXgb: Initial Success")


    @md_std_log()
    def fit(self, X, y, evals = [], save = "model"):
        bp, tp, cp = self.model_set_param()

        # evals = [(xgb.DMatrix(d[0], label = d[1], nthread = bp.get("n_jobs")), d[2]) for d in evals]
        if type(X) is dict:
            X = pd.DataFrame(X)
        train = xgb.DMatrix(X, label = y, nthread = bp.get("n_jobs"))

        if cp.get("nfold"):
            model = xgb.cv(bp, train, **cp)

            bst_eval = model.iloc[-1].to_dict()
            bst_eval["ntree"] = model.__len__()
            idx_key = [i for i in model.columns if i.startswith("test") and i.endswith("mean")][0]

            return bst_eval[idx_key],  bst_eval# evaluate result/ add train round to result

        self.model = xgb.train(bp, train, **tp)
        self.stat_varperf = pd.DataFrame([{'variable':k, 'xgb':v} for k,v in self.model.get_fscore().items()])

        if save:
            self.save(save = save)

        return sf.modeval_stat_index(self.model.predict(xgb.DMatrix(X)), y=y, target_type=self.target_type), { d[2]: sf.modeval_stat_index(self.model.predict(xgb.DMatrix(d[0]), y=d[1]), target_type=self.target_type)  for d in evals}

    @md_std_log()
    def predict(self, X, ntree_limit=0, pred_leaf=False):
        if type(X) is pd.DataFrame:
            X = xgb.DMatrix(X)
        return self.model.predict(X, ntree_limit = ntree_limit, pred_leaf = pred_leaf)

    @md_std_log()
    def save(self, save, model = None, *args, **kw):
        model = "obj_model_xgb" if model is None  else model
        if save in ["obj", "all"]:
            file_path = os.path.join(self.wkdir, "obj", model+".pkl")
            with open(file_path, 'wb') as f:
                cPickle.dump(self, f)

        if save:
            model = "model_xgb" if model is None else model
            file_path = os.path.join(self.wkdir, "model", model+".model")
            self.model.save_model(file_path)

    @md_std_log()
    def export(self, lang="pmml", model=None, *args, **kw):
        model = "model_xgb" if model is None else model
        # if lang=="pmml":
        #     file_path = os.path.join(self.wkdir, "model", model + ".model")
        #     self.model.save_model(file_path)

    def load(self, path = "model", model = None):
        model = "model_xgb" if model is None else model
        file_path = os.path.join(self.wkdir, path, model + ".model")
        self.model = xgb.Booster()
        self.model.load_model(file_path)

    @md_std_log()
    def varsele_get_perf(self):
        return self.stat_varperf

    def md_modelsele_imp(self):
        return self.stat_varperf

    @md_std_log()
    def model_set_param(self):
        """
        https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters
        :param kw:
        booster - gbtree, gblinear or dart
        verbosity - Valid values are 0 (silent), 1 (warning), 2 (info), 3 (debug).
        learning_rate - Step size shrinkage used in update to prevents overfitting
        gamma - Minimum loss reduction required to make a further partition on a leaf node of the tree.
        max_depth - Maximum depth of a tree.
        min_child_weight - Minimum sum of instance weight (hessian) needed in a child.
        subsample - Subsample ratio of the training instances.
        max_leaves - Maximum number of nodes to be added.

        :return:
        """
        booster_param = ["booster", "verbosity", "learning_rate", "gamma", "min_split_loss","max_depth", "min_child_weight", "subsample"
                  , "colsample_bytree", "colsample_bylevel", "colsample_bynode", "lambda", "alpha", "max_leaves", "objective"
                  , "base_score", "metric", "n_estimators", "n_jobs"]
        train_param = ["ntree", "obj", "feval", "maximize", "feval", "maximize", "early_stopping_rounds", "evals_result"
                       ,"verbose_eval"]
        cv_param = ["ntree", "cv", "stratified", "metrics", "obj", "feval", "early_stopping_rounds", "verbose_eval"]
        mapper = {"cv":"nfold", "ntree":"num_boost_round", "metric":"eval_metric"}
        obj_map = {"c":"reg:linear", "b":"binary:logistic", "p":"count:poisson", "s":"survival:cox", "m":"multi:softprob", "r":"rank:pairwise"}

        bp = {mapper.get(k) if mapper.get(k) else k : v for k,v in self.kw.items() if k in booster_param}
        tp = {mapper.get(k) if mapper.get(k) else k: v for k, v in self.kw.items() if k in train_param}
        cp = {mapper.get(k) if mapper.get(k) else k: v for k, v in self.kw.items() if k in cv_param}

        bp["objective"] = bp.get("objective") if bp.get("objective") else obj_map.get(self.target_type)

        return bp, tp, cp



    