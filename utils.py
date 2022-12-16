import numpy as np
import pandas as pd

def set_index(dataRaw):
    col_time = dataRaw.columns[0]
    dataRaw[col_time] = pd.to_datetime(dataRaw[col_time])
    dataRaw = dataRaw.set_index(col_time)
    return dataRaw

def cut_and_shuffle(ds=None, T=32, seed=42):
    N, C = ds.shape
    columns = ds.columns
    index = ds.index
    #for c in range(C):
    c = 0
    serie_name = columns[c]
    serie_indexs = index
    serie_values = ds.iloc[:,c].values
    samples_values = [serie_values[i: i + T] for i in range(0, N - T+1)] 
    samples_index = [serie_indexs[i: i + T] for i in range(0, N - T+1)] 
    samples_class = [serie_name for s in range(len(samples_values))] 

    return samples_values, samples_index, samples_class