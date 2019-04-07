import logging

import numpy as np
import pandas as pd
from sklearn.tree import _tree
from sklearn import tree

def quantile(arr, bin=10, miss_set=[np.nan], *args, **kw):
    """返回特定数组的分位数，在给定分位点和缺失值集合的情况下.

        Args:
            arr: 数组.
            bin: 待切分的分位段.
            miss_set: 缺失值集合.

        Returns:
            分位数数组. For  example:
            待补充

        Raises:

        """
    arr = arr[~np.in1d(arr, miss_set) & ~np.isnan(arr)]
    return np.percentile(arr, np.arange(0, 100, int(100 / bin))[1:])


def cut(arr, cuts, miss_set , upper = 1, lower = 0):

    if type(cuts) == type(0):
        cuts = list(quantile(arr, cuts, miss_set))

    result = pd.cut(arr, [lower] + cuts + [upper]).astype(str)

    return np.where(np.in1d(arr, miss_set), "Missing", result)

def cut2(arr, cuts, miss_set):
    """
    返回给定数组的cut排序，0为缺失
    :param arr: 待cut数组
    :param cuts: cut point
    :param miss_set: 缺失集合
    :return:
    """
    result = 1
    for p in cuts:
        result = result + (arr > p)
    return np.where(np.in1d(arr, miss_set), 0, result)

def woe_tbl(x, y, classes = [0, 1]):
    df = pd.DataFrame({"decile":x, "segment":y, "cnt":1})
    df_cnt = df.groupby(["decile", "segment"]).sum().reset_index()
    df_cnt["pct"] = df_cnt["cnt"]/df_cnt.groupby(["segment"])["cnt"].transform(sum)

    df_summary = df_cnt.pivot_table(index=['decile'], columns='segment', values='pct').reset_index()
    woe = np.log(df_summary[classes[1]]/df_summary[classes[0]])
    df_summary["woe"] = np.where(np.isnan(df_summary[classes[1]]), np.nanmin(woe), np.where(np.isnan(df_summary[classes[0]]), np.nanmax(woe), woe))
    iv = (df_summary[classes[1]] - df_summary[classes[0]]) * df_summary["woe"]
    df_summary["iv"] = np.where(np.isnan(iv) | np.isinf(iv), 0, iv)

    return df_summary


def obj_info(obj):
    """
    根据入参返回相应的对象信息
    :param obj: 入参
    :return: 对象信息
    """
    if type(obj) in [pd.DataFrame, np.ndarray, pd.Series]:
        return obj.shape
    if type(obj) is dict:
        return list(obj.keys())
    return obj


def is_numeric(obj):
    """
    返回array对象是否为数值型
    :param obj:
    :return:
    """
    return obj.dtype.kind not in "OSU"

def modeval_stat_index(pred, y, target_type = "b"):
    return modeval_stat_index_bin(pred, y)


def modeval_stat_index_bin(pred, y):
    '''
    need to add Somer’s D
    '''
    idx_sorted = np.argsort(pred)
    y_sorted = y[idx_sorted]
    # pred_sorted = pred[idx_sorted]
    cdf_pos = np.cumsum(y_sorted) / sum(y_sorted)
    cdf_neg = np.cumsum(1 - y_sorted) / sum(1 - y_sorted)
    KS = max(np.abs(cdf_pos - cdf_neg))

    integ_slice_X = cdf_neg[1:] - cdf_neg[:-1]
    integ_slice_Y = 1 - (cdf_pos[1:] + cdf_pos[:-1]) / 2
    AUC = sum(integ_slice_X * integ_slice_Y)
    GINI = 2 * AUC - 1
    return KS, AUC, GINI