import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from DataCleanUp import *


def solLplot():
    da = transform_data(pd.read_csv('data.csv', encoding='latin-1'))

    colnm = 'USD_hamP,USD_DjiP,auct_yr,measure_hwd_cm,unique,exec_pmortem,category'.split(',')
    sol = da.query("artist_name =='Sol LeWitt'")[colnm].sort_values(by=['category', 'auct_yr'])
    sol = sol[sol.auct_yr > 2000]
    g = sol.groupby(['category','auct_yr'])

    sol_AmeanCnt = sol.groupby(['auct_yr']).USD_DjiP.agg(['mean', 'count'])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for ky, grp in sol.groupby(['category']):
        ax.plot(g.auct_yr.first(), g.USD_DjiP.mean(), label=ky)
    ax.legend()
