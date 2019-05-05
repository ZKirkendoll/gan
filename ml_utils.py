#!/usr/bin/python
# coding=utf-8

# libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import inspect
import collections
from pprint import pprint as pprint
import string
import sys
import re
import operator
import random
import datetime
import math
import random as rnd

"""Preprocessor"""
T, F = 'T', 'F'
ML_UNIT_TEST = F
PRINT_DEBUG = T
ML_OUTPUT = T

pd.set_option("display.max_columns", 15)
desired_width = 360
pd.set_option('display.width', desired_width)

"""""""""
 Helper Functions
"""""""""


def df_explore(df):
    pprint(df.head())
    pprint(df.info())
    pprint(df.describe())
    pprint(df.columns)


def Paramdict():
    return collections.defaultdict(Paramdict)


"""""""""
Debug Functions
"""""""""


def ml_print(data):
    if PRINT_DEBUG is T:
        output = data
        pprint("(DEBUG) [%s] %s" % (inspect.stack()[1][3], output))


def ml_error(data):
    output = data
    print("(ERROR) [%s] %s" % (inspect.stack()[1][3], output))


def ml_output(data):
    if ML_OUTPUT is T:
        print("(INFO) Output: %s" % data)


def ml_result(data, duration):
    print("(INFO) Final Result: %s" % data)
    print("(INFO) Execution Time (ms): %s" % (duration))


"""""""""
 Common
"""""""""
