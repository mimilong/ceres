
from .MdBase import *
from .utils import *
import _pickle as cPickle
import pandas as pd
from hyperopt import fmin, tpe, Trials, STATUS_OK, space_eval
from .MdModelSelection.MdModelApi import mapping


# print(ms.__dict__)

class MdHyperopt(MdBase):
    def __init__(self, model, space, wkdir=".", minimize = True, logger=logging.getLogger('root'), indents=1,log_stack=[], *args, **kw):
        MdBase.__init__(self, logger, indents, log_stack)
        self.wkdir = wkdir
        self.method = model
        self.kw = kw
        self.space = space
        self.to_min = 1 if minimize else -1


        self.info("ModelTool {}: Initial Success".format(self.__class__.__name__))

    @md_std_log()
    def fit(self, X, y, max_evals=20, save = None, *args):
        objective = self.hyperopt_objfunc_factory(X = X, y=y, **self.kw)
        trials = Trials()
        best = fmin(objective, self.space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
        self.best = best

        # trial = [{k:v[0] for k, v in t['misc']['vals'].items()} for t in trials.trials]
        # evals = [t['result']['eval'] for t in trials.trials]

        stat_hyperopt = [{k:v[0] for k, v in t['misc']['vals'].items()} for t in trials.trials]
        stat_hyperopt = pd.DataFrame([dict(t, **space_eval(self.space, t)) for t in stat_hyperopt])
        self.stat_hyperopt = pd.concat([stat_hyperopt, pd.DataFrame([t['result']['eval'] for t in trials.trials])], axis=1)

        if save:
            self.save(save=save)

        return space_eval(self.space, best)

    @md_std_log()
    def save(self, save, *args, **kw):
        if save in ["obj", "all"]:
            file_path = os.path.join(self.wkdir, "obj", "obj_hyperopt.pkl")
            with open(file_path, 'wb') as f:
                cPickle.dump(self, f)

        if save in ["stat", "all"]:
            path = os.path.join(self.wkdir, "stat")
            self.stat_hyperopt.to_csv(path + "/stat_hyperopt.csv", index=False)

    @md_std_log()
    def hyperopt_objfunc_factory(self, X, y, evals=[], **kw):
        m = self.method
        to_min = self.to_min
        def objective(search):
            params = dict(kw, **search)
            model = mapping.get(m)(**params)
            metric, metric_eval = model.fit(X = X, y = y, evals = evals)
            return {'loss': metric * to_min, 'eval': metric_eval, 'status': STATUS_OK}
        return objective



