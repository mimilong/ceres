import pandas as pd
dft = pd.DataFrame([dict(variable = k, **v) for k,v in obj_processor.model_proc_num.items()])

np.max([0, 1,3, np.nan], )