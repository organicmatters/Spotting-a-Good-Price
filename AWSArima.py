import boto3
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class ArimaPipeline:
    
    def __init__(self):
        self.train = None
        self.test = None
        self.predictions = None
        self.dates = None
        self.failed_list = []
        self.model = None
        self.instance_name = None
    
    
    def apply_mask(self, dataframe:object, instance_types_pickle:object, instance_num:int):
        df = dataframe
        instance = instance_types_pickle
        instance_ts = instance_types_pickle[instance_num]
        self.instance_name = instance_ts
        print(instance_ts)
        mask = (df['AvailabilityZone'] == instance_ts[0]) & (df['InstanceType'] == instance_ts[1]) & (df['ProductDescription'] == instance_ts[2])
        price_series = df[mask]
        return price_series
    
    def downsample(self, instance_ts:object):
        prices = instance_ts[["SpotPrice"]]
        prices = prices.dropna()
        prices = prices.set_index(pd.DatetimeIndex(instance_ts.Timestamp))
        prices = prices.SpotPrice.resample('D').mean()
        self.dates = prices.index.strftime("%Y-%m-%d")
        return prices
    
    def train_test_split(self, time_series:object):
        # 80/20 train, test split
        size = int(len(time_series) * 0.80)
        self.train, self.test = time_series[0:size], time_series[size:len(time_series)]
        #return train, test
    
    def create_a_model(self, training_set:object):
        model = ARIMA(training_set, order=(10,1,0))
        self.model = model.fit()
        return self.model
    
    def forecast(self, forecast_len: int):
        self.predictions = self.model.forecast(forecast_len)[0]

    def calculate_error(self):
        error = mean_squared_error(self.test, self.predictions)
        return error
    
    def run_model(self, dataframe:object, instance_types_pickle:object, instance_num:int):
        ts = self.apply_mask(dataframe, instance_types_pickle, instance_num)
        ds = self.downsample(ts)
        self.train_test_split(ds)
        tsmodel = self.create_a_model(self.train)
        self.forecast(len(self.test))
        return self.calculate_error()
    
    def pipeline_iteration(self, dataframe:object, instance_types_pickle:object):
        
        total_error = 0
        for time_series_num in range(len(dataframe[0:20])):
            try:
                self.predictions = []
                ts = self.apply_mask(dataframe, instance_types_pickle, time_series_num)
                ds = self.downsample(ts)
                self.train_test_split(ds)
                tsmodel = self.create_a_model(self.train)
                self.forecast(len(self.test))
                error = self.calculate_error()
                total_error += error 
                print(time_series_num, error)
            except:
                print(time_series_num, "Error")
                self.failed_list.append(time_series_num)
                continue
        print(total_error)
    
    def plot_predictions(self):
        calc = len(self.train) - 1
        preds = list(np.full(calc, None)) +  list(self.train[-1:]) + list(self.predictions)
        actual = list(self.train) + list(self.test)
        #Uncomment the code below to not show the actual values for the test set on the plot
        #actual = list(self.train) + list(np.full(len(self.test), None))
        ticks_x = np.linspace(0,90,6)
        dates = [self.dates[int(ticks_x[0])], self.dates[int(ticks_x[1])], self.dates[int(ticks_x[2])], self.dates[int(ticks_x[3])], self.dates[int(ticks_x[4])], self.dates[-1]]
        plt.figure(figsize = [12,8])
        plt.plot(np.arange(len(self.train) + len(self.predictions)) , preds , 'purple', np.arange(len(self.train) + len(self.test)), actual, 'orange');
        plt.title(f'Forecasted vs Actual Spot Price of an Instance', fontsize = 20) # {self.instance_name[1]}
        plt.xticks(ticks_x, dates)
        #plt.ylim((3.5, 5))
        plt.ylabel('Price ($)', fontsize = 15)
        plt.legend(['Predicted', 'Actual'], fontsize = 'xx-large')
       
        return plt.show()