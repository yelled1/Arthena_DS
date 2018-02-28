import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm
#from sklearn import tree
import pydotplus
import graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score,mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelBinarizer
from pathlib import Path
from DataCleanUp import *
import forestci as fci
from RandForestRcls import modelRandClsR

def load_files(fnm='test.csv', naV=-999):
    dfTrain = transform_data(pd.read_pickle('raw_train.pickle'))
    if Path(fnm).is_file():
        test_data = pd.read_csv(fnm, encoding='latin-1')
    elif Path('test.pickle').is_file():
        test_data = pd.read_pickle('test.pickle')
    else:
        raise ValueError('test Data Error!')
    dfTest = transform_data(test_data)
    return dfTrain, dfTest

if __name__ == '__main__':
    naV=-999
    df_train, df_test = load_files()
    ycol   = ['USD_DjiP']
    colnms = 'auct_yr,measurement_height_cm,unique,artist_birth_year,exec_pmortem,USD_Hest'.split(',')
    encoder = LabelBinarizer()
    #y = df_train.hammer_price.fillna(self.naV)
    yf = df_train[ycol].fillna(naV).values.reshape((-1))
    xf = df_train.filter(colnms).fillna(naV).values
    Xf = np.concatenate((encoder.fit_transform(df_train.artist_name), xf), axis=1)
    yP = df_test[ycol].fillna(naV).values.reshape((-1))
    xP = df_test.filter(colnms).fillna(naV).values
    XP = np.concatenate((encoder.fit_transform(df_test.artist_name), xP), axis=1)

    M  = modelRandClsR({'max_depth':12, 'random_state':0, 'n_estimators':30})
    M.predict(Xf, yf, XP, yP, False)

    M.pred_ints(percentile=95)
    M.pred_int_calc(False)
    M.score_v
    M.plotYvsPredit()

    a=M.df.sort_values(['v']).reset_index()
    plt.errorbar(a.v,a.p_m,yerr=[a.p_d,a.p_u], fmt='--o')
    plt.scatter(a.p_m, a.v)

if 1:
    yColnms = ['USD_DjiP']
    xColnms = 'auct_yr,measurement_height_cm,unique,artist_birth_year,exec_pmortem,USD_Hest'.split(',')
    df_train = transform_data(pd.read_pickle('raw_train.pickle'))
    M = modelRandClsR()

    M.predict(Xf, yf, Xp, yp, False)
    M.pred_ints(percentile=75)
    M.pred_int_calc(True)
    M.score_v
    M.plotYvsPredit()

    M.predict(ret=False)
    M.pred_ints()
    M.pred_int_calc()
    M.score_v
    #plotYvsPredit(M) # this fails prob due to naV
    M.plotYvsPredit() # this fails prob due to naV
    M.df.columns
    M.df.co     

    trueV = M.yP
    df = pd.DataFrame()
    df['v']=trueV
    df['p_d']=M.err_dn
    trueV.shape
    len(M.err_dn)
    df['p_u']=self.err_up
    df['p_m']=self.err_mean

    trueV.shape
    M.err_dn.shape