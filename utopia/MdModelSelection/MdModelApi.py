import codecs
import inspect
from os.path import dirname
from importlib import import_module

from scipy.special import expit
import _pickle as cPickle

from ..MdBase import *
from ..utils import *
from .. import statfun as sf

mapping = {}
for p in os.listdir(os.path.abspath(dirname(__file__))):
    p = p.split(".")[0]
    if p !=  "MdModelApi" and p.startswith("MdModelSelection"):
        # print(__name__, p)
        mapping[p.split("MdModelSelection")[-1].lower()] = getattr(import_module("..{}".format(p), __name__), p)

# evals (X, y, ""segment")
class MdModelApi(MdBase):
    def __init__(self, wkdir=".", target_type='b', logger=logging.getLogger('root'), indents=1, log_stack=[], *args, **kw):
        '''
        '''
        MdBase.__init__(self, logger, indents, log_stack)
        self.wkdir = wkdir
        self.target_type = target_type
        self.kw = kw
        self.model = None

        self.info("ModelTool MdModelSelection: Initial Success")


    @md_std_log()
    def fit(self, X, y, subset, evals = []):
        self.model.fit(X = pd.DataFrame(X)[subset], y = y, evals = evals)
        self.stat_model_varperf = self.model.model_varimp()

    @md_std_log()
    def predict(self, X, **kw):
        return self.model.predict(X = X, **kw)

    @md_std_log()
    def varsele_get_perf(self):
        return self.varsele_get_perf

    @md_std_log()
    def save(self, save, *args, **kw):
        self.model.save(save, *args, **kw)

    @md_std_log()
    def export(self, lang="pmml", subset=None, *args, **kw):
        self.model.export(lang=lang, subset=subset, *args, **kw)


    