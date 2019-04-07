import logging

import numpy as np
import pandas as pd
from sklearn.tree import _tree
from sklearn import tree

from .statfun import *
from .MdBase import MdBase
from .utils import md_std_log


class MdUtilBinning(MdBase):
    def __init__(self, method = "optimal", miss_set=[np.nan], logger = logging.getLogger('root'), indents = 1, log_stack = [], *args, **kw):
        MdBase.__init__(self, logger, indents, log_stack)
        self.method = method
        self.miss_set = miss_set
        self.binner = quantile if method == "even" else MdUtilBinningOpt(logger = logger, indents = indents, log_stack = log_stack, *args, **kw)

        self.info("ModelTool MdUtilBinning: Initial Success")

    @md_std_log()
    def fit(self, x, y, *args, **kw):
        if self.method == "even":
            return self.binner(arr=x, miss_set=self.miss_set, **kw)
        return self.binner.fit(x = x, y = y)


class MdUtilBinningOpt(MdBase):
    def __init__(self, target_type='b', miss_set=[np.nan], logger = logging.getLogger('root'), indents = 1, log_stack = [], *args, **kw):
        MdBase.__init__(self, logger, indents, log_stack)
        self.binner = tree.DecisionTreeClassifier(**kw) if target_type == 'b' else tree.DecisionTreeRegressor(**kw)
        self.miss_set = miss_set

        self.info("ModelTool MdUtilBinningOpt: Initial Success")

    @md_std_log()
    def fit(self, x, y, *args, **kw):

        self.binner.fit(X = np.reshape(x, [-1,1]), y = y)
        return np.unique(self.binner.tree_.threshold[self.binner.tree_.feature != _tree.TREE_UNDEFINED])
