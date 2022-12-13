import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score
from prts import ts_precision, ts_recall
import pickle

def shift(serie, delay):
    output = serie.shift(delay)
    return output


class PeriodicMeanStd():
    def __init__(self, flag_day=False, flag_week=False, flag_hour=False, flag_minute=False,
    model_name="PeriodicMeanStd"):
        self.flag_day = flag_day
        self.flag_week = flag_week 
        self.flag_hour = flag_hour 
        self.flag_minute = flag_minute
        self.periodic_mean = pd.DataFrame()
        self.periodic_std = pd.DataFrame()
        self.conditions = []
        self.index_names = []
        self.name = model_name


    def fit(self, df=None):
        X = df.copy()

        if self.flag_day:
            self.conditions.append(X.index.day)
            self.index_names.append("day")
        if self.flag_week:
            self.conditions.append(X.index.weekday)
            self.index_names.append("weekday")
        if self.flag_hour:
            self.conditions.append(X.index.hour)
            self.index_names.append("hour")
        if self.flag_minute:
            self.conditions.append(X.index.minute)
            self.index_names.append("minute")

        self.periodic_mean = X.groupby(self.conditions).mean()
        self.periodic_mean.index = self.periodic_mean.index.set_names(self.index_names)
        self.periodic_std = X.groupby(self.conditions).std()
        self.periodic_std.index = self.periodic_std.index.set_names(self.index_names)

        self.periodic_mean.to_csv(self.name+"_mean.csv")
        self.periodic_std.to_csv(self.name+"_std.csv")


        return self


    def fit_transform(self, df=None):
        fit(df)
        output_mean = df.copy()
        output_std = df.copy()

        num = df.shape[0]
        for i in range(num):
            output_mean[i:i+1] = self.periodic_mean.loc[(self.conditions[0][i], self.conditions[1][i], self.conditions[2][i])]
            output_std[i:i+1] = self.periodic_std.loc[(self.conditions[0][i], self.conditions[1][i], self.conditions[2][i])]

        return output_mean, output_std


    def transform(self, df=None, load_model=False, path=None):
        output_mean = df.copy()
        output_std = df.copy()

        if load_model:
            if path != None:
                periodic_mean_aux = pd.read_csv(path+self.name+"_mean.csv")
                periodic_std_aux = pd.read_csv(path+self.name+"_std.csv")
            else:
                periodic_mean_aux = pd.read_csv(self.name+"_mean.csv")
                periodic_std_aux = pd.read_csv(self.name+"_std.csv")
        else:
            periodic_mean_aux = self.periodic_mean
            periodic_std_aux = self.periodic_std          

        num = df.shape[0]
        for i in range(num):
            output_mean[i:i+1] = periodic_mean_aux.loc[(self.conditions[0][i], self.conditions[1][i], self.conditions[2][i])]
            output_std[i:i+1] = periodic_std_aux.loc[(self.conditions[0][i], self.conditions[1][i], self.conditions[2][i])]

        return output_mean, output_std


    def alpha_selection(self, df=None, df_y=None, load_model=False, path=None,
                           custom_metrics=False, al=0, cardinality='reciprocal',
                           bias='front', max_alpha=7):           
        # Model
        if load_model:
            if path != None:
                periodic_mean_aux = pd.read_csv(path+self.name+"_mean.csv")
                periodic_std_aux = pd.read_csv(path+self.name+"_std.csv")
            else:
                periodic_mean_aux = pd.read_csv(self.name+"_mean.csv")
                periodic_std_aux = pd.read_csv(self.name+"_std.csv")
        else:
            periodic_mean_aux = self.periodic_mean
            periodic_std_aux = self.periodic_std   

         # Condiotions of df
        conditions_aux = []
        if self.flag_day:
            conditions_aux.append(df.index.day)
            self.index_names.append("day")
        if self.flag_week:
            conditions_aux.append(df.index.weekday)
            self.index_names.append("weekday")
        if self.flag_hour:
            conditions_aux.append(df.index.hour)
            self.index_names.append("hour")
        if self.flag_minute:
            conditions_aux.append(df.index.minute)
            self.index_names.append("minute")

        output_mean = df.copy()
        output_std = df.copy()
        num = df.shape[0]
        for i in range(num):
            output_mean[i:i+1] = periodic_mean_aux.loc[(conditions_aux[0][i], conditions_aux[1][i], conditions_aux[2][i])]
            output_std[i:i+1] = periodic_std_aux.loc[(conditions_aux[0][i], conditions_aux[1][i], conditions_aux[2][i])]

        # Data
        X = df.values
        y = df_y.values
            
        print('Alpha selection...')
        best_f1 = np.zeros(df.shape[1])
        best_alpha = max_alpha*np.ones(df.shape[1])

        for alpha in np.arange(max_alpha, 1, -1):
            thdown = output_mean - alpha*output_std
            thup = output_mean + alpha*output_std
            
            self.pre_predict = (df < thdown) | (df > thup)
            self.pre_predict = self.pre_predict.astype(int)
            
            for c in range(df.shape[1]):
                df_y_col = df_y.iloc[:,c].values
                pre_predict_col = self.pre_predict.iloc[:,c].values

                if custom_metrics:
                    if np.allclose(np.unique(pre_predict_col), np.array([0, 1])) or np.allclose(np.unique(pre_predict_col), np.array([1])):
                        pre_value = ts_precision(df_y_col, pre_predict_col, 
                                        al, cardinality, bias)
                        rec_value = ts_recall(df_y_col, pre_predict_col, 
                                        al, cardinality, bias)
                        f1_value = 2*(pre_value*rec_value)/(pre_value+rec_value+1e-6)
                    else:
                        pre_value = 0
                        rec_value = 0
                        f1_value = 0
                else:
                    f1_value = f1_score(df_y_col, pre_predict_col, pos_label=1)
                    pre_value = precision_score(df_y_col, pre_predict_col, pos_label=1)
                    rec_value = recall_score(df_y_col, pre_predict_col, pos_label=1)
                
                if f1_value >= best_f1[c]:
                    best_f1[c] = f1_value
                    best_alpha[c] = alpha

        self.alpha = best_alpha
        self.f1_val = best_f1
        
        with open(self.name + '_alpha.pkl', 'wb') as f:
            pickle.dump(best_alpha, f)
            f.close()
        
        return self


    def predict(self, df, load_model=False, load_alpha=False, path=None):
        # Model
        if load_model:
            if path != None:
                periodic_mean_aux = pd.read_csv(path+self.name+"_mean.csv")
                periodic_std_aux = pd.read_csv(path+self.name+"_std.csv")
            else:
                periodic_mean_aux = pd.read_csv(self.name+"_mean.csv")
                periodic_std_aux = pd.read_csv(self.name+"_std.csv")
        else:
            periodic_mean_aux = self.periodic_mean
            periodic_std_aux = self.periodic_std 

         # Condiotions of df
        conditions_aux = []
        if self.flag_day:
            conditions_aux.append(df.index.day)
            self.index_names.append("day")
        if self.flag_week:
            conditions_aux.append(df.index.weekday)
            self.index_names.append("weekday")
        if self.flag_hour:
            conditions_aux.append(df.index.hour)
            self.index_names.append("hour")
        if self.flag_minute:
            conditions_aux.append(df.index.minute)
            self.index_names.append("minute")


        output_mean = df.copy()
        output_std = df.copy()
        num = df.shape[0]
        for i in range(num):
            output_mean[i:i+1] = periodic_mean_aux.loc[(conditions_aux[0][i], conditions_aux[1][i], conditions_aux[2][i])]
            output_std[i:i+1] = periodic_std_aux.loc[(conditions_aux[0][i], conditions_aux[1][i], conditions_aux[2][i])]


        if load_alpha:
            with open(self.name + '_alpha.pkl', 'rb') as f:
                alpha = pickle.load(f)
                f.close()
        else:
            alpha = self.alpha

        thdown = output_mean - alpha*output_std
        thup = output_mean + alpha*output_std
        
        detections = (df < thdown) | (df > thup)
        detections = detections.astype(int)

        return detections, output_mean, thdown, thup

        


    



        
    