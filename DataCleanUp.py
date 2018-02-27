import re, numbers
import numpy as np
import pandas as pd
from functools import partial #currency -> USD?
from sklearn.model_selection import train_test_split

#Converts Currencies
def currency_conv(curr):
    if   curr == "GBP": return 1.4
    elif curr == "EUR": return 1.24
    elif curr == "USD": return 1.24
    else: return np.NaN

# Mainly used for execution Yr Data
def ckInt(is_int, tf=False, dbg=False, btwnYrs=True):
    if is_int != is_int: return np.NaN
    elif isinstance(is_int, numbers.Number):
        if dbg: print("dbg: numeric")
        if tf: return True
        else:
            #year of execution Not ridiculous
            if btwnYrs and (is_int < 1800 or is_int > 2019): return np.NaN
            return is_int
    else:
        if dbg: print("dbg: Notnum")
        if tf: return False
        else:
            r = re.search("[1-9][0-9][0-9][0-9]", is_int)
            if not(r): return np.NaN
            else:
                nYr = float(r.group(0))
                if btwnYrs and (nYr < 1800 or nYr > 2019): return np.NaN
                return nYr
# Using partial functions for quick apply to columnal data
TFInt = partial(ckInt,True, True)
TFint = partial(ckInt,True, False)

def binny(df_series, bins=25):
    bins = np.linspace(df_series.min(), df_series.max(), bins)
    grps = df_series.groupby(np.digitize(df_series, bins))
    return grps.agg(['min', 'max','count'])

def cnv_mean(x): return float(x.replace(',',''))

def djIndexSeries():
    #Adding Dji Avg Data Given Acution years goes back to 80's. This will be used as asset value normalizer
    dji = pd.read_csv("DJind.txt", sep="\t")
    dji['DJlevel'] = dji.ValueClose.apply(cnv_mean)
    return dji.groupby(['Year'], as_index=False).agg({'DJlevel': 'mean'})

def transform_data(data):
    # Adding currency coversion rates
    data['CONVrate'] = data.currency.apply(currency_conv)
    # Mark if work is unique == more $$$ than others
    data['unique'] = data.edition == "unique"
    #auction date post death? or near/at death = death increase price
    data['auct_yr'] =  pd.DatetimeIndex(data.auction_date).year
    data['exec_pmortem'] = data.auct_yr >= data.artist_death_year
    data["yr_exec"] = data.year_of_execution.apply(ckInt)
    DJi = djIndexSeries()
    df = data.merge(DJi, how='left', left_on=['auct_yr'], right_on=['Year'])
    del df['Year']
    ## Getting rid of useless Data
    df = df[~np.isnan(df.hammer_price)]
    df = df[df.hammer_price > 1000]; df.shape
    # ## Adding USD converated & Dow Jones Adjusted Data
    # ### Seriously Data is from 80's & all assets should be normalized
    df['USD_hamP'] = df.hammer_price * df.CONVrate
    df['USD_Hest'] = df.estimate_high * df.CONVrate
    df['USD_Lest'] = df.estimate_low * df.CONVrate
    df['USD_DjiP'] = df.USD_hamP / df.DJlevel
    df['USD_Dest'] = (df.USD_Hest+df.USD_Lest) / (df.DJlevel*2)
    df['USD_estH'] = df.USD_Hest / df.DJlevel
    df['measure_hwd_cm'] = df.measurement_height_cm + df.measurement_width_cm + df.measurement_depth_cm
    return df

if __name__ == '__main__':
    # Reading the input Data in
    rawdata = pd.read_csv('data.csv', encoding='latin-1')
    raw_train, raw_test = train_test_split(rawdata, test_size=.2)
    raw_train.to_pickle('raw_train.pickle')
    raw_test.to_pickle('test.pickle')
    #df_train = transform_data(raw_train)
    #print(df_train.shape)
    #df_train.head()
