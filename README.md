# utopia 使用文档

*utopia* 是针对机器学习建模流程的自动化工具，位于<https://github.com/mimilong/ceres>

## 1. 概述

- 主要支持的功能为：
  - 自动化日志配置 ```utopia.set_logger```

  - 数据预处理， 及中间结果的存储：   ```utopia.MdProcess```

  - 特征评估及选择， 及中间结果的存储： ```utopia.mv.MdVarSelectApi```

  - 特征变化， 及中间结果的存储： ```utopia.vt.MdVarTransApi```

  - 参数调优， 及中间结果的存储：```utopia.MdHyperopt```

  - 模型训练， 及中间结果的存储： ```utopia.ms.MdModelSelectionXgb```

## 2. 目录规范

```bash
└── project
    ├── config                        <- [ 训练配置文件目录 ]
    │   └── config.config  
    │
    ├── data                          <- [ 数据集文件夹 ]
    │   ├── dataset_model             	<- [ 预处理后存储的数据集 ]
    │   ├── dataset_model_preproc     	<- [ 预处理后存储的数据集 ]
    │   ├── dataset_model_subset      	<- [ 变量筛选后的数据集 ]
    │   └── dataset_model_trans       	<- [ 变量变化后存储的数据集 ]
    │
    ├── log                           <- [ 日志文件目录 ]
    │   ├── postfix-error.log         	<- [ info日志 ]
    │   └── postfix-run.log           	<- [ error日志 ]
    │
    ├── model                         <- [ 模型文件 ]
    │   ├── model_proc_cat            	<- [ 预处理MdProcess.类别型变量预处理生成代码 ]
    │   ├── model_proc_num            	<- [ 预处理MdProcess.数值型变量预处理生成代码 ]
    │   ├── model_trans_mrf_001       	<- [ 变量变换MdVarTransApi.模型变换文件（如：叶子节点做新特征的变化代码） ]
    │   ├── model_trans_norm          	<- [ 变量变换MdVarTransApi.数据标准化代码 ]
    │   ├── model_trans_power         	<- [ 变量变换MdVarTransApi.常规变换生成代码 ]
    │   ├── model_trans_woe           	<- [ 变量变换MdVarTransApi.WOE变换生成代码 ]
    |   └── model                       <- [ 模型训练MdModelSelectionXgb.模型文件 ]
    │    
    ├── obj                           <- [ 类实例存储 ]
    │  
    ├── report                        <- [ 模型报告目录 ]
    │
    ├── source                        <- [ 模型训练代码 ]
    │   ├── 01.data_prepare
    │   ├── 02.train
    │   └── 03.evaluate
    │
    └── stat                          <- [ 变量统计文件，EDA及标准化前导出 ]
        ├── stat_eda_cat.csv          	<- [ 预处理MdProcess.类别型变量EDA ]
        ├── stat_eda_num.csv          	<- [ 预处理MdProcess.数值型变量EDA ]
        ├── stat_eda_trans.csv        	<- [ 变换后变量EDA ]
        ├── stat_hyperopt.csv         	<- [ 参数调优MdHyperopt.参数调优后的过程展示 ]
        └── stat_varperf.csv          	<- [ 特征选择MdVarSelectApi.变量评估表现 ]

```



## 3. 代码准备

- 导入工具包Demo

```python
# 1 基本配置项
## 1.1 导基本包
import pandas as pd
from hyperopt import hp
import numpy as np

## 1.2 配置外部包和数据路径
## 1.3 加载分析通用工具包(环境路径上一步添加)
import utopia as mt
## 1.4 加载项目配置相关项
## 1.5 配置日志项
## 1.6 其他分析相关配置
miss_set = [999]
target = 'y'

```



- 日志设置
  - ```utopia.set_logger```
  - 主要参数说明
    - log_theme ： string ; default = ""; 日志生成的文件前缀
    - path： string; default = None;  日志存储为文件路径， 默认为工作路径下
  - Demo

```python
## 1.5 配置日志项
log_theme = "tool-test"
mt.set_logger(postfix=log_theme, logerr="error", loginfo="run", path='log', logname=None)
logger = logging.getLogger('root')

```





- 以下Demo将基于定义的数据集：

```python
# 2. load data
dataset_model = pd.read_csv('dataset_train.csv')
variable = np.load('variable.npy')

# 3. split train and test
dataset_train_X = dataset_model[variable]
dataset_train_y = np.array(dataset_model[target])
```



# 一、预处理 - MdProcess

## 1. 方法说明

```utopia.MdProcess()``` 新建数据预处理实例，并配置预处理的方式方法。

## 2. 主要参数

| 参数名称     | 参数类型 | 默认值                               | 解释                                                     |
| ------------ | -------- | ------------------------------------ | -------------------------------------------------------- |
| var_type     | dict     | None                                 | 变量类型字典,示例 {"var1":"numeric", "var2":"nominal"}   |
| var_desc     | dict     | None                                 | 变量解释字典, 示例 {"var1":"年龄"}，用于描述变量中文含义 |
| miss_set     | list     | [None, np.nan, 999999999, 999999990] | 缺失值枚举集合                                           |
| miss_impute  | string   | median                               | 缺失值填充方法 ，支持 ：['median','mode','mean']         |
| q_trim       | float    | 0.01                                 | floor cap对应的百分数，在0-1之间                         |
| pct_indicate | float    | 0.05                                 | 添加标识列所应到的百分比                                 |
| cat_proc     | string   | None                                 | 类别型变量预处理                                         |

## 3. 主要函数

- ```MdProcess.fit(X, y)```  对输入数据集划分为数值型、类别性特征， 分别批量进行统计分析(下面称**EDA**)，并存储统计结果

  - 连续型特征统计指标
    - n、nmss、min、max、mean、median、mode、q01、q99、pct_miss、pct_mode、pct_q01、pct_q99、std、trim_mean、trim_std、
  - 类别性统计指标
    - n、mean、std、rank、prefix

- ```MdProcess.predict(X)```利用对数据的统计分析结果，进行预处理，**返回： DataFrame ,预处理后的数据集** 包括以下步骤：

  1. 对缺失值进行填充。

  2. 对缺失率超过指定比率的特征， 增加新列进行标记。

     即：【新列为1时表示缺失、为0时表示有值】；新列在原有特征名基础上增加前缀：```FMS_```

  3. 认为小于```q_trim```处值及大于$$1-q_trim$$处值的是异常值； 对异常值进行平滑处理， 即： 【小于```q_trim```处值的， 用```q_trim```处值填充； 大于$$1- q_trim$$处值的， 用$$1 - q_trim$$处值填充】

  - 对异常比例超过pct_indicate的， 增加新列进行标记。 新列在原有特征名基础上增加前缀：```FFP_```、```FCP_```
  - 对类别性特征进行预处理。预处理方式为```cat_proc```传入的方法； 新列在原有特征名基础上增加前缀：```FXX_``` ；

- ```MdProcess.view()``` 如果使用了fit函数，当调用本函数时返回：``` [stat_eda_num, stat_eda_cat], var_desc```

- ```MdProcess.load()```  可以在生成EDA文件后， 直接导入对应变量， 不需要再调用```MdProcess.fit(X, y)```  

- ```MdProcess.export()``` 存储整个实例



## 4. Demo


```python
# 4. eda & preprocess
obj_processor = mt.MdProcess(miss_set = miss_set, cat_proc = "onehot")
obj_processor.fit(dataset_train_X, dataset_train_y, save = "stat")
## obj_processor.load() # 如果已生成EDA文件， 可以不使用fit函数的情况下，执行下面语句
obj_processor.export("json")
_, var_desc = obj_processor.view()
dataset_train_proc = obj_processor.predict(dataset_train_X)

```



# 二、特征选择-MdVarSelectApi

## 1. 方法说明

​       ``` utopia.mv.MdVarSelectApi()```集成了多种特征评估方法， 并生成基于多方法的特征评估结果文件， 该借口可扩展， 便于添加新的评估手段。 目前集成的评估方法为： IV计算、Xgboost特征排序两种特种筛选方法。

​        其中：Xgboost特征排序，对指定参数训练一个简单模型， 通过模型中特征的importance, 对其重要性进行评估。 这里将支持对特征随机分组后， 分别进行排序—— 解决： 训练的模型，由于树棵数较少等原因，导致的大部分特征没有选入模型， 从而没有重要性分数的问题。

​       代码依赖：

 -  ```utopia.vt.MdVarTransWoe```
 -  ```utopia.ms.MdModelSelectionXgb```

## 2. 主要参数

| 参数名称 | 参数类型 | 默认值                     | 解释                                                         |
| -------- | -------- | -------------------------- | ------------------------------------------------------------ |
| selector | dict     | {"IV":{"threshold": 0.01}} | Key:  特征评估方法, 包括：IV、Xgb；<br>.   value:  dict ;{'threshold'：n, 'top': m}, 其中n,m是数值: <br>                threshold：特征评估指标>n的所有特征<br>                top: 特征排序的前m个特征 |
| groups   | int      | 1                          | Xgboost特征评估时， 对特征分组的组数                         |
| **param  | dict     | None                       | 其他参数<br>如xgboost参数                                    |


## 3. 主要函数

- ```utopia.mv.MdVarSelectApi.fit(X, y)```  

  - 生成特征评估结果文件， 
  - 返回方法```stat_varperf```以便查看全部评估排序结果, 并存储该结果于```stat/stat_varperf.csv```
  - 返回方法 ```model_varsele_vars```  ， 所有符合评估标准的重要特征列表, 并存储该结果于```model/model_varsele_vars.json```

  ```stat_varperf```结果示例：

  | variable                           | xg_score | xg_order | iv_score    | iv_order |
  | ---------------------------------- | -------- | -------- | ----------- | -------- |
  | rto2_num_1dcp6mwwwt_num_1dcp6mwwwc | 2299     | 4        | 0.251171876 | 0        |
  | rto2_num_1dcp4mwwwt_num_1dcp4mwwwc | 2019     | 6        | 0.248752001 | 1        |
  | rto2_num_1dcp3mwwwt_num_1dcp3mwwwc | 2116     | 5        | 0.238307152 | 2        |

- ```utopia.mv.MdVarSelectApi.predict(X,subset)``` 

  - **返回符合筛选条件的特征列表**
  - 参数```subset``` list  ；default = None ;需要删除的特征 ；

- ```utopia.mv.MdVarSelectApi.view()``` 

  - 返回 ```model_varsele_vars, None```

- ```utopia.mv.MdVarSelectApi.load()```  

  - 不需要执行```fit```函数， 直接导入特征评估结果及符合评估标准的重要特征列表



## 5. Demo

```python
# 特征筛选参数
selector = {"IV":{"threshold": 0.01}, "Xgb":{"top":5}}
params = {"groups":2, "ntree":50, "learning_rate":0.1, "splitter":"optimal","max_depth":4}


obj_selector = mt.mv.MdVarSelectApi(target_type = "b", selector = selector, **params) # 初始化
# obj_selector.varsele_get_perf()
obj_selector.fit(X = dataset_train_proc, y = dataset_train_y, save = "stat") # 特征评估
obj_selector.export(lang = "json") # 存储对象
dataset_train_sele = obj_selector.predict(dataset_train_proc) # 重要特征列表
```



# 四、变量变换-MdVarTransApi

## 1. 方法说明

​       ``` utopia.vt.MdVarTransApi()```集成了多种特征变化方法， 并生成基于多方法的特征变化的对应解释， 该接口可扩展， 便于添加新的变化方法。 目前集成的变化方法为： 特征归一化、特征数值变化、数据离散化且使用WOE值标记

 - 特征归一化 ``` utopia.vt.MdVarTransNorm()```：$$(variable\_value - mean)/std​$$ ; 
 - 特征数值变化``` utopia.vt.MdVarTransPower()```：
    - SQ： $$ x^2 ​$$
    - SR：$$\sqrt{x}​$$
    - IV：$$ /frac{1}{x+1}​$$
    - LN：$$ln(1+x)$$
 - 特征离散化及使用WOE标记``` utopia.vt.MdVarTransWoe()```
   - 通过决策树对特征分箱， 计算WOE， 使用WOE值对特征离散化



## 2. 主要参数

| 参数名称 | 参数类型 | 默认值                | 解释                                                 |
| -------- | -------- | --------------------- | ---------------------------------------------------- |
| miss_set | list     | [np.nan]              | 缺失值列表                                           |
| flag_set | list     | ["FMS", "FFP", "FCP"] | 特殊标记的特征列， 不做特征变化                      |
| mapper   | Dict     | {"woe":{}}            | 数值变化的方法选择；{"woe":{}, "power":{},"norm":{}} |



## 3. 主要函数

- ```utopia.vt.MdVarTransApi.fit(X, y)```  
  - 为特征变化做运算准备， eg: 
    - 标准化时， 对统计指标的计算；
    - 数值变化时，准备对连续特征的记录； 
    - 离散化时， WOE的计算及存储结果
- ```utopia.vt.MdVarTransApi.predict(X)``` **返回：  DataFrame ； 输出数值变化后的结果文件**
- ```utopia.vt.MdVarTransApi.view()```  **返回： stat_vartrans_woe, var_desc**
- ```utopia.vt.MdVarTransApi.load()```  
  - 不需要执行```fit```函数， 直接导入特征变化需要的运算准备





## 5. Demo

```python
# 6. 变量变换
mapper = {"woe":{}, "power":{}}
params = {"flag_set" :["FMS", "FFP", "FCP", "FX"], "splitter":"optimal","max_depth":4}

obj_transformer = mt.vt.MdVarTransApi(target_type = "b", mapper = mapper, **params)
obj_transformer.fit(X = dataset_train_proc, y = dataset_train_y, save = "stat")

obj_transformer.export(lang = "json")

dataset_train_trans = obj_transformer.predict(dataset_train_sele)
```







# 五、参数调优-MdHyperopt

## 1. 方法说明

```utopia.MdHyperopt())``` 使用贝叶斯优化方法， 以AUC为评估指标， 寻找最使AUC最优的超参数组

依赖类：```utopia.MdModelSelection.MdModelApi```

## 2. 主要参数

| 参数名称     | 参数类型 | 默认值 | 解释                                          |
| ------------ | -------- | ------ | --------------------------------------------- |
| model        | string   | 必选   | 模型类型； eg： xgb lr                        |
| space        | Dict     | 必选   | 参数选择空间；                                |
| minimize     | Bool     | True   | 损失函数是否是单调下降                        |
| 其他模型参数 | dict     |        | 出入的最后参数， 形式是： **{}， 其他模型参数 |



## 3. 主要函数

- ```utopia.MdHyperopt.fit(X, y,max_evals)```
  - ```max_evals```: 参数组的最大探索数
  - **返回最优参数组**
- ```utopia.MdHyperopt.save(X)```





## 5. Demo

```python
# 7. 超参数调优
fspace = {"learning_rate":hp.loguniform("learning_rate", np.log(10**-5), 0),
          "max_depth":hp.choice("max_depth", [2,3,4])}

kw = {"objective":"binary:logistic", "metric":"auc", "cv":5, "ntree":200, "early_stopping_rounds":5}

obj_optimizer = mt.MdHyperopt(model = "xgb", space = fspace
                              ,  minimize = False, target_type = "b", **kw)
obj_optimizer.fit(X = dataset_train_proc, y = dataset_train_y, max_evals=5)

obj_optimizer.save(save = "stat")
```







# 六、模型选择及训练-MdModelSelectionXgb

## 1. 方法说明

```utopia.ms.MdModelSelection``` 集成多个模型进行训练、存储、预测及预测结果的评估。 目前支持lr、xgboost两种建模方法， KS、AUC、gini三种评估指标

- ```utopia.ms.MdModelSelectionXgb```  训练、存储、预测xgboost模型
- ```utopia.ms.MdModelSelectionLR```  训练、存储、预测LR模型



## 2. 主要参数

| 参数名称    | 参数类型 | 默认值 | 解释                                                         |
| ----------- | -------- | ------ | ------------------------------------------------------------ |
| target_type | String   | b      | 目标类型；b: 二分分类问题                                    |
| kw          | Dict     |        | 模型训练参数, 其中重要参数<br>nfold : 训练模型时， 是否采用K折验证， 寻找最优的树棵树 |



## 3. 主要函数

- ```utopia.ms.MdModelSelectionXgb.fit(X, y, evals = [])```   

  - ```evals```  在预测时， 对传入数据集进行预测及评估， 常用于的测试数据集评估， 以展现训练模型效果
  - 训练模型并存储模型， 预测并输出 KS、AUC、gini三种评估指标的值

- ```utopia.ms.MdModelSelectionXgb.save(save)```存储训练模型

  - ```save```存储文件名称

- ```utopia.ms.MdModelSelectionXgb.predict(X)``` 对传入数据集进行预测

- ```utopia.ms.MdModelSelectionXgb.load()```  导入已生成的模型

  

## 4. Demo

```python
# 8. 模型训练与评估
obj_predictor = mt.ms.MdModelSelectionXgb()
obj_predictor.fit(dataset_train_proc, dataset_train_y)

obj_predictor.save(save = "model")
```

