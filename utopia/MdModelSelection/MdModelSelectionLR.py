import codecs
import itertools

from scipy.special import expit
import _pickle as cPickle

import sklearn.linear_model as glm

from ..MdBase import *
from ..utils import *
from .. import statfun as sf


class MdModelSelectionLR(MdBase):
    def __init__(self, wkdir=".", target_type='b', logger=logging.getLogger('root'), indents=1, log_stack=[], *args, **kw):
        """

        :param wkdir:
        :param target_type:
        :param logger:
        :param indents:
        :param log_stack:
        :param args:
        :param kw:
        使用glmnet API的参数
        family=c("gaussian","binomial","poisson","multinomial","cox","mgaussian")
        bj_map = {"c":"reg:linear", "b":"binary:logistic", "p":"count:poisson", "s":"survival:cox", "m":"multi:softprob", "r":"rank:pairwise"}
        alpha lambda intercept thresh
        https://pypi.org/project/glmnet-python/

        https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
        linear_model.LogisticRegressionCV(Cs=10, fit_intercept=True, cv=’warn’, dual=False, penalty=’l2’, scoring=None,
        solver=’lbfgs’, tol=0.0001, max_iter=100, class_weight=None, n_jobs=None, verbose=0, refit=True, intercept_scaling=1.0,
        multi_class=’warn’, random_state=None)

        linear_model.ElasticNetCV(l1_ratio=0.5, eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, normalize=False,
        precompute=’auto’, max_iter=1000, tol=0.0001, cv=’warn’, copy_X=True, verbose=0, n_jobs=None, positive=False, random_state=None, selection=’cyclic’)
        """
        MdBase.__init__(self, logger, indents, log_stack)
        self.wkdir = wkdir
        self.target_type = target_type
        self.kw = kw
        self.model = self.md_modelsele_get_obj()
        self.metric = kw.get("metric")

        self.info("ModelTool MdModelSelectionLR: Initial Success")

    @md_std_log()
    def fit(self, X, y, save = "stat", evals = []):
        X = pd.DataFrame(X)
        self.vars_all = list(X.columns)
        self.info(self.model.get_params())

        self.model.fit(X = X, y = y)
        self.md_modelsele_imp()
        self.save(save = save)

        # eval metric/loss
        metric = sf.modeval_stat_index(pred=self.predict(X = X), y= y, target_type=self.target_type, metric = self.metric)
        eval_metric = {d[2]: sf.modeval_stat_index(pred=self.predict(X = d[0]), y=d[1], target_type=self.target_type) for d in evals}
        eval_metric = [{"{}-{}".format(dn, idx):val for idx,val in idxs.items()} for dn, idxs in eval_metric.items()]
        eval_metric = dict(itertools.chain(*map(dict.items, eval_metric)))

        eval_metric["{}-{}".format("train", self.metric)] = metric
        return metric, eval_metric


    @md_std_log()
    def predict(self, X, **kw):
        if isinstance(X, pd.DataFrame):
            X = {k: np.array(X[k]) for k in set(self.model_coef.keys()) & set(X.columns)}
        X["intercept"] = 1 # 改变了原数据集
        result = np.array(0)
        for k,v in self.model_coef.items():
            result = result + X[k] * v

        if self.target_type == "b":
            result = expit(result)

        return result

    @md_std_log()
    def varsele_get_perf(self):
        return self.stat_varperf

    def md_modelsele_imp(self):
        self.stat_varperf = pd.DataFrame({"variable":self.vars_all + ["intercept"], "coef":np.append(self.model.coef_, self.model.intercept_)})
        self.model_coef = self.stat_varperf[self.stat_varperf["coef"] != 0].set_index("variable")["coef"].to_dict()

    @md_std_log()
    def save(self, save, model=None, *args, **kw):
        model = "obj_{}".format("model_lr" if model is None else model)
        if save in ["obj", "all"]:
            file_path = os.path.join(self.wkdir, "obj", model + ".pkl")
            with open(file_path, 'wb') as f:
                cPickle.dump(self, f)

        if save:
            model = "stat_{}.csv".format("model_lr" if model is None else model)
            file_path = os.path.join(self.wkdir, "stat", model)
            self.stat_varperf.to_csv(file_path, index=False)

    @md_std_log()
    def export(self, lang="pmml", model=None, *args, **kw):
        model = "model_lr" if model is None else model
        if lang == "json":
            file_path = os.path.join(self.wkdir, "model", model + ".json")
            with codecs.open(file_path, 'w', 'utf-8') as f:
                json.dump(self.model_coef, f, ensure_ascii=False)

    @md_std_log()
    def load(self, path = "model",lang="pmml",  model = None):
        model = "model_lr" if model is None else model
        if lang=="json":
            file_path = os.path.join(self.wkdir, path, model + ".json")

            with codecs.open(file_path) as f:
                self.model_coef = json.load(f)

    @md_std_log()
    def md_modelsele_get_obj(self):
        """
        alpha lambda intercept thresh
        family=c("gaussian","binomial","poisson","multinomial","cox","mgaussian")
        bj_map = {"c":"reg:linear", "b":"binary:logistic", "p":"count:poisson", "s":"survival:cox", "m":"multi:softprob", "r":"rank:pairwise"}
        :return:
        """
        pub_param = ["verbose", "n_jobs","random_state", "copy_X", "solver", "max_iter"]
        linear_map = {"intercept":"fit_intercept", "lambda":"alpha", "alpha":"l1_ratio", "thresh":"tol"}
        cv_linear_map = {"intercept":"fit_intercept", "lambda":"alphas", "alpha":"l1_ratio", "thresh":"tol"}
        log_map = {"intercept":"fit_intercept", "thresh":"tol"}

        params = {k:v for k,v in self.kw.items() if k in pub_param}

        if self.target_type == "c":
            if self.kw.get("cv"):
                params = dict(params, **{cv_linear_map[k]:v for k,v in self.kw.items() if k in cv_linear_map})
                return glm.ElasticNetCV(**params)
            params = dict(params, **{linear_map[k]: v for k, v in self.kw.items() if k in linear_map})
            return glm.LinearRegression(**params)

        if self.target_type == "b":
            params = dict(params, **{log_map[k]: v for k, v in self.kw.items() if k in log_map})
            params["penalty"] = "l1" if self.kw.get("alpha", 1) == 1 else "l2"
            if self.kw.get("cv"):
                lambdas = self.kw.get("lambda", 1)
                lambdas = lambdas if isinstance(lambdas, list) else [lambdas]
                params["Cs"] = list(1/np.array(lambdas))
                return glm.LogisticRegressionCV(cv = self.kw["cv"], **params)

            params["C"] = 1/self.kw.get("lambda", 1)
            return glm.LogisticRegression(**params)






    