from ..MdBase import *
from ..MdModelSelection.MdModelSelectionXgb import MdModelSelectionXgb
from ..utils import *
from .. import statfun as sf

class MdVarSelectXgb(MdBase):
    def __init__(self, wkdir=".", seed = 0, groups = 1, target_type = "b", logger=logging.getLogger('root'), indents=1, log_stack=[], *args, **kw):
        '''
        '''
        MdBase.__init__(self, logger, indents, log_stack)
        self.wkdir = wkdir
        self.target_type = target_type
        self.seed = seed
        self.groups = groups
        self.kw = kw

        self.info("ModelTool MdVarSelectXgb: Initial Success")

    def fit(self, X, y, *args, **kw):
        vars_all = list(X.keys())
        np.random.shuffle(vars_all)
        vars_p_round = int(len(vars_all)/self.groups)
        X = pd.DataFrame(self.md_varsele_preproc(X = X, y =y))
        predictor = MdModelSelectionXgb(wkdir=self.wkdir, seed=self.seed, target_type=self.target_type, logger=self.logger
                                         ,indents=self.indents, log_stack=self.log_stack, **self.kw)
        ns = 0
        stat_varperf = []
        while ns < len(vars_all):
            predictor.fit(X = X[vars_all[ns:(ns+vars_p_round)]], y = y, save=None)
            stat_varperf.append(predictor.varsele_get_perf())
            ns = ns+vars_p_round

        self.stat_varperf = pd.concat(stat_varperf)


    @md_std_log()
    def md_varsele_preproc(self, X, y):
        return {k: v if sf.is_numeric(v) else self.md_varsele_2num(x = v, y =y)  for k,v in X.items()}

    @md_std_log()
    def md_varsele_2num(self, x, y):
        df_stat = pd.DataFrame(dict(n=y, mean=y, std=y, value=x)).groupby(['value'], as_index=False).agg(
            {'n': len, 'mean': np.mean, 'std': lambda x: np.std(x, ddof=1)})
        df_stat["rank"] = np.argsort(df_stat["mean"])
        str2num = df_stat.set_index("value")["rank"].to_dict()
        result = 0
        for k, v in str2num.items():
            result = result + (x == k)*v

        return result

    @md_std_log()
    def varsele_get_perf(self):
        return self.stat_varperf

