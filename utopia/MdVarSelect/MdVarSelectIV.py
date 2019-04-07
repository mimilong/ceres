from ..MdBase import *
from ..MdVarTrans.MdVarTransWoe import MdVarTransWoe
from ..utils import *


class MdVarSelectIV(MdBase):
    def __init__(self, target_type, logger = logging.getLogger('root'), indents = 1, log_stack = [], *args, **kw):
        MdBase.__init__(self, logger, indents, log_stack)
        self.target_type = target_type
        self.transformer = MdVarTransWoe(logger = logger, indents = indents, log_stack = log_stack, *args, **kw)

        self.info("ModelTool MdVarSelectIV: Initial Success")

    @md_std_log()
    def fit(self, X, y, *args, **kw):
        self.transformer.fit(X = X, y = y)
        stat_vartrans_woe, _ = self.transformer.view()
        self.stat_varperf = stat_vartrans_woe.groupby(["variable"])[["iv"]].sum()

    @md_std_log()
    def varsele_get_perf(self):
        return self.stat_varperf
