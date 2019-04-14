from .MdBase import *
from .utils import *
import _pickle as cPickle

from scipy import stats
import codecs
import json
import re


class MdProcess(MdBase):
    def __init__(self, wkdir=".", var_type=None, var_desc=None,
                 miss_set=[None, np.nan, 999999999, 999999990],
                 miss_impute='median', q_trim=0.01, pct_indicate=0.05, cat_proc=None, logger=logging.getLogger('root'), indents = 1, log_stack = [],  *args, **kw):
        '''
        :param wkdir工作路径，整体模型项目路径
        :param logger ：日志对象，用于打印模型训练中间状态
        :param var_type 变量类型字典{"var1":"numeric", "var2":"nominal"}，默认值为None
        :param var_desc:变量解释字典{"var1":"年龄"}，用于描述变量意义的字典
        :param miss_set 缺失值枚举集合
        :param miss_impute 确实值填充方法['median','mode','mean', 0]，默认median
        :param q_trim floor cap对应的百分数
        :param pct_indicate 添加标识列所应到的百分比
        :param cat_proc类别型变量预处理
        '''
        MdBase.__init__(self, logger, indents, log_stack)
        self.var_type = var_type
        self.var_desc = var_desc
        self.miss_set = miss_set
        self.miss_impute = miss_impute  # 缺失值填充方式['median','mode','mean']，默认median
        self.q_trim = q_trim
        self.pct_indicate = pct_indicate
        self.cat_proc = cat_proc
        self.wkdir = wkdir

        self.logger.info("{}[{}] - {}".format(self.indents*"\t", "-".join(self.log_stack), "ModelTool MdProcess: Initial Success"))

    @md_std_log()
    def fit(self, X, y, save="stat", *args, **kw):

        if self.var_type is None:
            self.get_var_type(X = X)
        if self.var_desc is None:
            self.get_var_desc(X = X)

        X_num, X_cat = self.split_by_type(X = X)
        self.eda_num(X = X_num)
        self.eda_cat(X = X_cat, y = y)
        self.preproc_cat_dummy()
        self.update_varinfo()

        self.save(save = save)


    @md_std_log()
    def get_var_type(self, X):
        """
        获取入参数据集的数据类型，根据对应的数据类型将数据映射成numeric binary ordinal nominal几个数据类型, 待补充完善
        """
        type_all = set(X.get_dtype_counts().index)
        type_num = [t for t in type_all if re.compile('float|int').match(t)]
        # type_fact = [t for t in type_all if re.compile('object|category|bool').match(t)]

        var_type = {name: 'numeric' if t in type_num else 'nominal'
                    for name, t in X.dtypes.astype(str).to_dict().items()}
        self.var_type = var_type


    @md_std_log()
    def get_var_desc(self, X):
        """
        获取入参数据集的变量名构造变量描述名
        """
        variables = X.columns
        self.var_desc = {c: c for c in variables}

    @md_std_log()
    def split_by_type(self, X):
        vars_num = [k for k, v in self.var_type.items() if v in ["numeric", "binary"]]
        vars_cat = [k for k, v in self.var_type.items() if v in ["nominal", "ordinal"]]

        self.vars_num = vars_num
        self.vars_cat = vars_cat

        return X[vars_num], X[vars_cat]

    @md_std_log()
    def eda_num_base(self, arr):
        '''
        arr: numpy array as input
        output: dict with all statistics
        '''

        result = {}
        n = len(arr)
        result['n'] = n
        # n nmss min max mean median mode qnn pct_miss pct_mode pct_qnn std trim_mean trim_std
        arr_nonmiss = arr[~(np.isnan(arr) | np.in1d(arr, self.miss_set))]
        n_nonmiss = len(arr_nonmiss)

        result['nmiss'] = n - n_nonmiss
        result['pct_miss'] = result['nmiss'] / result['n']

        result['min'] = np.nan if n_nonmiss == 0 else min(arr_nonmiss)
        result['max'] = np.nan if n_nonmiss == 0 else max(arr_nonmiss)
        result['mean'] = np.nan if n_nonmiss == 0 else np.mean(arr_nonmiss)
        result['std'] = np.nan if n_nonmiss == 0 else np.std(arr_nonmiss, ddof=1)
        # calculate mode
        result['mode'] = np.nan if n_nonmiss == 0 else stats.mode(arr_nonmiss)[0][0]

        quant_point = [1, 5, 25, 50, 75, 95, 99]
        quant_arr = [np.nan] * len(quant_point) if n_nonmiss == 0 else np.percentile(arr_nonmiss, quant_point,
                                                                                     interpolation='lower')

        quant_dict = dict(zip(['q%02d' % i for i in quant_point], quant_arr))

        result = dict(result, **quant_dict)

        result['pct_mode'] = sum(arr_nonmiss == result['mode']) / n
        result['pct_q01'] = sum(arr_nonmiss <= quant_dict['q01']) / n
        result['pct_q99'] = sum(arr_nonmiss >= quant_dict['q99']) / n

        # floor & cap treatment
        arr_trim = np.clip(arr_nonmiss, quant_dict['q01'], quant_dict['q99'])
        result['trim_mean'] = np.nan if n_nonmiss == 0 else np.mean(arr_trim)
        result['trim_std'] = np.nan if n_nonmiss == 0 else np.std(arr_trim, ddof=1)
        result['median'] = result['q50']
        return result

    @md_std_log()
    def eda_num(self, X):
        num_dict = {k: np.array(X[k], dtype=np.float64) for k in X.columns}
        eda_num_dict = [dict(self.eda_num_base(arr = v), variable=k) for k, v in num_dict.items()]

        self.stat_eda_num = pd.DataFrame(eda_num_dict)
        self.model_proc_num = self.stat_eda_num.set_index('variable').to_dict('index')

    @md_std_log()
    def eda_cat_base(self, x, y):
        '''
        y is  0/1 or continous data
        '''
        x = x.astype('str')
        return pd.DataFrame(dict(n=y, mean=y, std=y, value=x)).groupby(['value'], as_index=False).agg(
            {'n': len, 'mean': np.mean, 'std': lambda x: np.std(x, ddof=1)})


    @md_std_log()
    def eda_cat(self, X, y):
        cat_dict = {k: np.array(X[k]) for k in X.columns}
        self.stat_eda_cat = pd.concat([self.eda_cat_base(x=v, y=y).assign(variable = k) for k, v in cat_dict.items()])

    @md_std_log()
    def preproc_cat_dummy(self, prefix = "FX"):
        """
        哑变量标记处理函数
        :return:
        """
        if self.cat_proc is None:
            self.model_proc_cat = None

        elif self.cat_proc == "onehot" and self.stat_eda_cat is not None:
            df_eda_cat = self.stat_eda_cat.copy()
            df_eda_cat['rank'] = df_eda_cat.groupby(["variable"])['n'].rank(method = 'first', ascending = False)
            df_eda_cat['prefix'] = [prefix + '%02d' %i for i in df_eda_cat['rank']]
            self.stat_eda_cat = df_eda_cat

            self.model_proc_cat = df_eda_cat.groupby(["variable"]).apply(lambda x: dict(zip(x["value"], x["prefix"]))).to_dict()

    @md_std_log()
    def update_varinfo(self):
        '''
        更新变量描述信息
        var_desc: variable name, origin name, desc
        '''
        # origin part
        var_desc = {k:{"origin":k, "desc": self.var_desc[k], "type": v} for k,v in self.var_type.items()}

        # numeric tag part
        ## FMS_
        desc_FMS = {"FMS_{}".format(k):{"origin":k, "desc":"Flag of missing for {}".format(k), "type": "binary"} for k,v in self.model_proc_num.items() if v['pct_miss'] > self.pct_indicate and v['pct_miss'] < (1 - self.pct_indicate)}

        # FFP_
        desc_FFP = {"FFP_{}".format(k): {"origin": k, "desc": "Flag of floor point for {}".format(k), "type": "binary"} for k, v in
                    self.model_proc_num.items() if v['pct_q01'] > self.pct_indicate}

        # FCP_
        desc_FCP = {"FCP_{}".format(k): {"origin": k, "desc": "Flag of cap point for {}".format(k), "type": "binary"} for k, v in
                    self.model_proc_num.items() if v['pct_q99'] > self.pct_indicate}

        desc_FXX = self.update_varinfo_cat() if self.cat_proc else {}

        # FXX_
        var_desc = dict(dict(dict(var_desc, **desc_FCP), **desc_FMS), **desc_FFP)

        self.var_desc = dict(var_desc, **desc_FXX)

    @md_std_log()
    def update_varinfo_cat(self):
        if self.cat_proc == "onehot":
            desc_FXX = {}
            for var in self.vars_cat:
                map = self.model_proc_cat[var]
                desc_FXX = dict(desc_FXX, **{"{}_{}".format(v, var): {"origin": var, "desc": "Flag of {} for {}".format(k, var), "type": "binary"} for k, v in map.items()})
            return desc_FXX


    @md_std_log()
    def predict(self, X, subset = None, *args, **kw):
        '''
        逐个执行处理代码，对于具体变量为先做缺失值标记 填充 再做floor cap 最后做众数及高占比数据标记
        '''
        if type(X) is pd.DataFrame:
            X = {k: np.array(X[k]) for k in X.columns}

        vars_num = self.vars_num if subset is None else list(set(self.vars_num) & set(subset))
        vars_cat = self.vars_cat if subset is None else set(self.vars_cat) & set(subset)

        res = self.preproc_num_miss(X, subset = vars_num)  # 缺失值处理
        res = self.preproc_num_trim(res, subset = vars_num)  # floor cap 处理
        res = self.preproc_num_indicator(res, subset = vars_num)  #

        if self.cat_proc:
            res_cat = self.preproc_cat(X, subset = vars_cat)  # 类别型变量预处理
            return dict(res, **res_cat)

        return dict(res, **{X[k] for k in vars_cat})

    @md_std_log()
    def preproc_num_miss(self, X, subset = None):
        '''
                缺失值的标记
                缺失值的填充，所有的信息在stat_eda_num中获取，以及init函数中初始化的
        '''
        # 自定义缺失值标准化
        vars_flag = {v[4:] for v in self.var_desc.keys() if v[:3] == "FMS"}
        vars_flag = list(set(subset) & vars_flag)

        X_FMS = {"FMS_{}".format(k): (np.isnan(X[k]) | np.in1d(X[k], self.miss_set) + 0) for k in vars_flag}

        # 缺失值填充
        self.logger.debug(subset)
        X = {k:np.where(np.isnan(X[k]) | np.in1d(X[k], self.miss_set), self.model_proc_num[k][self.miss_impute], X[k]) for k in subset if self.model_proc_num[k]["pct_miss"] < (1 - self.pct_indicate)}

        return dict(X, **X_FMS)

    @md_std_log()
    def preproc_num_trim(self, X, subset):
        '''
            只做一件事
                对数据做floor cap的处理
        '''
        subset = set(subset) & set(X.keys())
        for var in subset:
            X[var] = np.clip(X[var], self.model_proc_num[var]['q%02d' % (self.q_trim*100)], self.model_proc_num[var]['q%02d' % ((1-self.q_trim)*100)])

        return X

    @md_std_log()
    def preproc_num_indicator(self, X, subset):
        '''
            只做一件事
                对mode lower bound 与 upper bound做标记
        '''
        subset = subset if subset else set(self.var_desc.keys())
        flag_FC = {var: desc["origin"] for var, desc in self.var_desc.items() if var[:3] in ('FFP', 'FCP') if var in subset}
        # flag_FC = set(subset) & flag_FC

        X_FC = {
            var:
                (X[origin] < self.model_proc_num[origin]['q%02d' % (self.pct_indicate * 100)]) + 0
                if var[:3] == 'FFP'
                else
                (X[origin] > self.model_proc_num[origin]['q%02d' % (100 - self.pct_indicate * 100)]) + 0
            for var, origin in flag_FC.items()}

        return dict(X, **X_FC)

    @md_std_log()
    def preproc_cat(self, X, subset):
        vars_orgin = subset # self.vars_cat if subset is None else list(set(subset) & set(self.vars_cat))
        if self.cat_proc == "onehot":
            result = {}
            for var in vars_orgin:
                map = self.model_proc_cat[var]
                result = dict(result, **{"{}_{}".format(v, var): (X[var] == k) + 0 for k,v in map.items()})
            return result


    def view(self):
        if not hasattr(self, "stat_eda_num"):
            self.stat_eda_num = pd.DataFrame([dict(variable = k, **v) for k,v in self.model_proc_num.items()])

        if not hasattr(self, "stat_eda_cat"):
            stat_eda_cat = []
            for k,v in self.model_proc_num.items():
                stat_eda_cat = stat_eda_cat + [{"variable":v, "value":val, "prefix":prefix} for val, prefix in v.items()]
            self.stat_eda_cat = pd.DataFrame(stat_eda_cat)

        return [self.stat_eda_num, self.stat_eda_cat], self.var_desc

    @md_std_log()
    def save(self, save, *args, **kw):
        if save in ["obj", "all"]:
            file_path = os.path.join(self.wkdir, "obj", "obj_proc.pkl")
            with open(file_path, 'wb') as f:
                cPickle.dump(self, f)

        if save in ["stat", "all"]:
            path = os.path.join(self.wkdir, "stat")
            if self.stat_eda_num is not None:
                self.stat_eda_num.to_csv(path + "/stat_eda_num.csv", index=False)
            if self.stat_eda_cat is not None:
                self.stat_eda_cat.to_csv(path + "/stat_eda_cat.csv", index=False)

    @md_std_log()
    def export(self, lang="pmml", subset=None, *args, **kw):
        '''只是导出代码，可以选择性导出部分代码
        通过lang路由到不同的生成代码函数中去，现在只写python就可以了
        '''
        path = os.path.join(self.wkdir, "model")
        if lang == "pmml":
            pass

        elif lang == 'json':
            with codecs.open('{}/{}'.format(path, 'model_eda_num.json'), 'w', 'utf-8') as f:
                json.dump(self.model_proc_num, f, ensure_ascii=False)

            with codecs.open('{}/{}'.format(path, 'model_eda_cat.json'), 'w', 'utf-8') as f:
                json.dump(self.model_proc_cat, f, ensure_ascii=False)

            with codecs.open('{}/{}'.format(path, 'model_proc_desc.json'), 'w', 'utf-8') as f:
                json.dump(self.var_desc, f, ensure_ascii=False)



    def load(self, path = "model"):
        with codecs.open('{}/{}'.format(path, 'model_eda_num.json')) as f:
            self.model_proc_num = json.load(f)

        with codecs.open('{}/{}'.format(path, 'model_eda_cat.json')) as f:
            self.model_proc_cat = json.load(f)

        with codecs.open('{}/{}'.format(path, 'model_proc_desc.json')) as f:
            self.var_desc = json.load(f)

        self.vars_num = list(self.model_proc_num.keys())
        self.vars_cat = list(self.model_proc_cat.keys())