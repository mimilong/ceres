# 1 基本配置项
## 1.1 导基本包
import sys
import os
import configparser
import logging
import json
import re

import numpy as np
import pandas as pd
import datetime

## 1.2 配置外部包和数据路径
tool_path = '/Users/long.li/Documents/work/project/data-analytics-utopia'
sys.path.append(tool_path)

## 1.3 加载分析通用工具包(环境路径上一步添加)
import utopia as mt
# import source.MdVarSelect as vs
from utopia import sf
import utopia.tool as tf

## 1.4 加载项目配置相关项
conf = configparser.ConfigParser()
conf.read('conf/config.ini', encoding="utf-8")
conf_ssh = tf.pub_geti_from_conf(conf, 'ssh')
conf_db = tf.pub_geti_from_conf(conf, 'postgre')

## 1.5 配置日志项
log_theme = "tool-test"
mt.set_logger(postfix=log_theme, logerr="error", loginfo="run", path='log', logname=None)
logger = logging.getLogger('root')

## 1.6 其他分析相关配置
### https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
target = 'y'
miss_set = [999]
split = "split"
vars_remove = []
vars_rm_sp = []