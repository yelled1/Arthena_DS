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

# Linear Ols regression for testing
#result = sm.ols(formula="USD_hamP ~ artist_name + auct_yr + measurement_height_cm + unique +artist_birth_year + exec_pmortem + USD_Hest ", data=df_train.fillna(self.naV)).fit()
#print(result.summary2())
#print(result.params)

class modelit:
    def __init__(self, fnm='test.csv', naV=-999):
        self.df_train = transform_data(pd.read_pickle('raw_train.pickle'))
        if Path(fnm).is_file():
            test_data = pd.read_csv(fnm, encoding='latin-1')
        elif Path('test.pickle').is_file():
            test_data = pd.read_pickle('test.pickle')
        else:
            raise ValueError('test Data Error!')
        self.df_test = transform_data(test_data)
        self.naV = naV

    def predict(self, colnms = None, ret=True):
        if not colnms:
            colnms = 'auct_yr,measurement_height_cm,unique,artist_birth_year,exec_pmortem,USD_Hest'.split(',')
        encoder = LabelBinarizer()
        #y = df_train.hammer_price.fillna(self.naV)
        self.y = self.df_train.USD_DjiP.fillna(self.naV)
        x = self.df_train.filter(colnms).fillna(self.naV).values
        self.X = np.concatenate((encoder.fit_transform(self.df_train.artist_name), x), axis=1)
        self.yP = self.df_test.USD_DjiP.fillna(self.naV)
        xP = self.df_test.filter(colnms).fillna(self.naV).values
        self.XP = np.concatenate((encoder.fit_transform(self.df_test.artist_name), xP), axis=1)

        self.clf = RandomForestRegressor(max_depth=12, random_state=0, n_estimators=30)
        self.clf.fit(self.X, self.y.values)
        self.score_v = self.clf.score(self.X,self.y.values)
        self.pred = self.clf.predict(self.XP)
        if ret: return self.pred

    def plotYvsPredit(self):
        self.mse = mean_squared_error(self.yP.values, self.pred)
        self.mae = mean_absolute_error(self.yP.values, self.pred)
        fig, ax = plt.subplots()
        ax.scatter(self.yP, self.pred, edgecolors=(0, 0, 0))
        ax.plot([self.yP.min(), self.yP.max()], [self.yP.min(), self.yP.max()], 'k--', lw=4)
        self.V_IJ_unbiased = fci.random_forest_error(self.clf, self.X, self.XP)
        ax.errorbar(self.yP, self.pred, yerr=np.sqrt(self.V_IJ_unbiased), fmt='o')
        ax.set_xlabel('Observed')
        ax.set_ylabel('Predicted')
        plt.show()

if __name__ == '__main__':
    M = modelit()
    M.predict(ret=False)
    M.score_v
    M.plotYvsPredit() # this fails prob due to naV

    M = modelit(naV=0)
    M.predict(ret=False)
    M.score_v
    M.plotYvsPredit() # this fails prob due to naV
