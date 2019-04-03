import boto3
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA


class PredictFuturePrice:
    
    def __init__(self):
        self.prices = {}
        self.forecasted_values = {}
        self.configurations = pd.read_pickle('all_instance_type_permutations.pkl')
        self.configs_to_run = []
        self.date_range = None
        
    def get_similar_instances(self, os_type, instance_type):
        '''This method returns all of the other regions of the same instance'''
        configs_to_run = []
        for config in self.configurations:
            if os_type in config and instance_type in config:
                configs_to_run.append(config)
        return configs_to_run
    
    def get_prices(self, start, end, instance, region):    
        client=boto3.client('ec2',region_name=region)
        prices=client.describe_spot_price_history(StartTime=start, EndTime=end, InstanceTypes=instance)  
        spot_price_datetimes = []
        spot_prices = []
        price_list = []
        
        for price in prices['SpotPriceHistory']:
            d = {'InstanceType': price['InstanceType'], 'AvailabilityZone': price['AvailabilityZone'], 'ProductDescription': price['ProductDescription'], 'SpotPrice': price['SpotPrice'], 'Timestamp': price['Timestamp']}
            price_list.append(d)
        
        df = pd.DataFrame(data=price_list)
        #df = df.set_index(pd.to_datetime(spot_price_datetimes))
        self.prices = df
        return df
    
    def downsample(self, instance_ts:object):
        prices = instance_ts.dropna()
        prices = prices[0].resample('D').mean()
        self.date_range = prices.index
        return prices
    
    def create_model(self, training_set:object):
        model = ARIMA(training_set, order=(10,1,0))
        model = model.fit()
        return model
    
    def multi_price(self, start, end, instance, os_type, region):
        configs_to_run = self.get_similar_instances(os_type=os_type, instance_type=instance)
        for config in configs_to_run: 
            client=boto3.client('ec2',region_name=config[0][:-1])
            instance_list = []
            os_type_list = []
            instance_list.append(instance)
            os_type_list.append(os_type)
            df_regions = pd.read_pickle('all_instance_type_permutations.pkl')
            prices=client.describe_spot_price_history(StartTime=start, EndTime=end, InstanceTypes=instance_list, ProductDescriptions=os_type_list, AvailabilityZone=config[0])       
            spot_price_datetimes = []
            spot_prices = []

            for price in prices['SpotPriceHistory']:
                spot_price_datetimes.append(price['Timestamp'])
                spot_prices.append(float(price['SpotPrice']))
            
            df = pd.DataFrame(spot_prices)
            df = df.set_index(pd.to_datetime(spot_price_datetimes))
            df = self.downsample(df)
            self.prices[config[0]] = df
    
    def multipipeline(self, num_days_forecast):
        
        for price_set in self.prices:
            try:
                model = self.create_model(self.prices[price_set])
                self.forecasted_values[price_set] = model.forecast(num_days_forecast)[0]
            except:
                self.forecasted_values[price_set] = np.full(num_days_forecast, self.prices[price_set][-1])

                