# 1 基本配置项
## 1.1 导基本包
import sys
import configparser
import logging
import numpy as np
import pandas as pd

tool_path = '/Users/long.li/Documents/work/project/data-analytics-utopia'
sys.path.append(tool_path)

## 1.3 加载分析通用工具包(环境路径上一步添加)
import utopia as mt
import utopia.statfun as sf
import utopia.tool as tf

## 1.5 配置日志项
log_theme = "tool-test-predict"
mt.set_logger(postfix=log_theme, logerr="error", loginfo="run", path='log', logname=None)
logger = logging.getLogger('root')

## 1.6 其他分析相关配置
### https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
target = 'y'
miss_set = [999]

########################################################
# 2. load data
dataset_raw = pd.read_csv("data/dataset_model.csv")
y = dataset_raw[target]

########################################################
# 3. preprocess
###
obj_processor = mt.MdProcess(miss_set = miss_set, cat_proc = "onehot")
obj_processor.load()
dataset_raw = obj_processor.predict(dataset_raw)
#
########################################################

# 4. 变量选择
selector = {"IV":{"threshold": 0.01}, "Xgb":{"top":5}}
params = {"groups":2, "ntree":50, "learning_rate":0.1, "splitter":"optimal","max_depth":4}


obj_selector = mt.mv.MdVarSelectApi(target_type = "b", selector = selector, **params)
obj_selector.load(path = "model")
dataset_raw = obj_selector.predict(dataset_raw)
len(dataset_raw.keys()) # 29

########################################################
# 5. 变量变换
mapper = {"woe":{}, "power":{}}
params = {"flag_set" :["FMS", "FFP", "FCP", "FX"], "splitter":"optimal","max_depth":4}

obj_transformer = mt.vt.MdVarTransApi(target_type = "b", mapper = mapper, **params)
obj_transformer.load(lang="json")

result = obj_transformer.predict(X = dataset_raw)
result.keys()
########################################################


########################################################
# 6. 模型预测
# 列转行，新的列名为decile
obj_predictor = mt.ms.MdModelSelectionXgb()
obj_predictor.load(lang="json")
result = obj_predictor.predict(X = dataset_raw)


