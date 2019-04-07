# 1 基本配置项
## 1.1 导基本包
import sys
import configparser
import logging
import json

import numpy as np

## 1.2 配置外部包和数据路径
tool_path = '/Users/long.li/Documents/work/project/model_tool'
sys.path.append(tool_path)

## 1.3 加载分析通用工具包(环境路径上一步添加)
import source as mt
import public_func.public as pf

## 1.4 加载项目配置相关项
conf = configparser.ConfigParser()
conf.read('conf/conf.ini', encoding="utf-8")
conf_ssh = pf.pub_geti_from_conf(conf, 'ssh')
conf_db = pf.pub_geti_from_conf(conf, 'postgre')

## 1.5 配置日志项
log_theme = "external-datasource-eval"
mt.set_logger(postfix=log_theme, logerr="error", loginfo="run", path='log', logname=None)
logger = logging.getLogger('root')

## 1.6 其他分析相关配置
target = "y"
vars_remove = []

########################################################
# 2. 加载sql并导数
with open('source/20.pull_model_data.sql', encoding='utf-8') as f:
	sql = json.loads(f.read())

# df = pd.read_csv('data/raw.csv')
dataset_model = pf.load_data_through_ssh(sql,conf_db,conf_ssh)
# dataset_model = pd.read_csv("data/dataset_model.csv", index_col = 0)
dataset_model["y"] = (dataset_model["overdue_day_1"] > 7) + 0
# dataset_model = dataset_model[1:5000]
dataset_model["split"] = np.where(np.random.uniform(size=4999) < 0.3, "test", "train")

dataset_model.to_csv("data/dataset_model.csv", index=False)



