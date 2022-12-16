import numpy as np
import pandas as pd


def set_index(dataRaw):
    col_time = dataRaw.columns[0]
    dataRaw[col_time] = pd.to_datetime(dataRaw[col_time])
    dataRaw = dataRaw.set_index(col_time)
    return dataRaw


def MTS2UTS(ds=None, T=32, seed=42):
    N, C = ds.shape
    columns = ds.columns
    index = ds.index
    
    samples_index = []
    samples_values = []
    samples_class = []
    for c in range(C):
        serie_index = index
        serie_values = ds.iloc[:,c].values
        serie_name = columns[c]
        samples_aux_index = [serie_index[i: i + T] for i in range(0, N - T+1)]
        samples_aux_values = [serie_values[i: i + T] for i in range(0, N - T+1)]  
        samples_aux_class = [serie_name for s in range(len(samples_aux_values))]
        samples_index += samples_aux_index
        samples_values += samples_aux_values
        samples_class += samples_aux_class

    return samples_values, samples_index, samples_class

