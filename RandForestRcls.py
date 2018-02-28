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
from sklearn.datasets import load_boston

# Linear Ols regression for testing
#result = sm.ols(formula="USD_hamP ~ artist_name + auct_yr + measurement_height_cm + unique +artist_birth_year + exec_pmortem + USD_Hest ", data=df_train.fillna(ModeModel.naV)).fit()
#print(result.summary2())
#print(result.params)

class modelRandClsR:
    def __init__(self, kargs):
        self.naV = -999
        self.clf = RandomForestRegressor(**kargs)

    def predict(self, X, y, XP, yP, ret=True):
        self.X, self.y, self.XP, self.yP = X, y, XP, yP
        self.clf.fit(self.X, self.y)
        self.score_v = self.clf.score(self.X,self.y)
        self.pred = self.clf.predict(self.XP)
        if ret: return self.pred

    def pred_ints(self, percentile=95):
        self.err_dn = []
        self.err_up = []
        self.err_mean = []
        self.predsA = []
        for x in range(len(self.XP)):
            preds = []
            for pred in self.clf.estimators_: preds.append(pred.predict(self.XP[x].reshape(1,-1))[0])
            self.err_dn.append(np.percentile(preds, (100 - percentile) / 2. ))
            self.err_up.append(np.percentile(preds, 100 - (100 - percentile) / 2.))
            self.err_mean.append(np.mean(preds))
            #self.predsA.append(preds)

    def pred_int_calc(self, calcV_IJ=False):
        trueV = self.yP
        self.df = pd.DataFrame()
        self.df['v']=trueV
        if calcV_IJ:
            self.df['V_IJ_unbiased'] = fci.random_forest_error(self.clf, self.X, self.XP)
        self.df['p_d']=self.err_dn
        self.df['p_u']=self.err_up
        self.df['p_m']=self.err_mean
        incorrect = ((np.sum(self.df.v > self.df.p_u) + np.sum(self.df.v < self.df.p_d)) / self.df.shape[0])
        return 1 - incorrect

    def plotYvsPredit(self):
        self.mse = mean_squared_error(self.yP, self.pred)
        self.mae = mean_absolute_error(self.yP, self.pred)
        a=self.df.sort_values(['v']).reset_index()
        plt.errorbar(a.v,a.p_m,yerr=[a.p_d,a.p_u], fmt='--o')
        #plt.scatter(a.v,a.p_m,color='green')
        #plt.errorbar(a.v, a.p_m, yerr=np.sqrt(self.V_IJ_unbiased), fmt='o')
        #plt.xlabel('Observed')


if __name__ == '__main__':
    boston = load_boston()
    X = boston["data"]
    Y = boston["target"]
    size = len(boston["data"])
    trainsize = 400
    idx = list(range(size))
    np.random.shuffle(idx)
    Xf, yf = X[idx[:trainsize]], Y[idx[:trainsize]]
    Xp, yp = X[idx[trainsize:]], Y[idx[trainsize:]]

    M  = modelRandClsR({'n_estimators':100, 'min_samples_leaf':1})
    M.predict(Xf, yf, Xp, yp, False)
    M.pred_ints(percentile=75)
    M.pred_int_calc(True)
    M.score_v
    M.plotYvsPredit()

    M.df.shape

    M.df.columns
    M.df.head()
    M.df.index
