# minimal example from:
# http://flask.pocoo.org/docs/quickstart/

from flask import Flask, request, render_template
#from get_AWS_prices import get_prices
from PricePredict import PredictFuturePrice
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np

app = Flask(__name__)

# creates an association between the / page and the entry_page function (defaults to GET)
@app.route('/')
def entry_page():
    return render_template('index.html')

@app.route('/predict_prices/', methods=['GET', 'POST'])
def render_message():

    # user-entered instance settings
    instance_parameters = ['instance_type', 'start_date', 'end_date', 'operating_system_type', 'region', 'forecast_length']

    # error messages to ensure correct ec2 parameters 
    messages = ["Please select an instance type.",
                "Please select a start date.",
                "Please select an end date.",
                "Please select an operating system type.",
                "Please select a region.",
                "Please select a forecast length."]
    
    inputs = []

    for i, inst in enumerate(instance_parameters):
        user_input = request.form[inst]
        if user_input == '':
            return render_template('index.html', message=messages[i])
        inputs.append(user_input)
   
    instance_pricing = PredictFuturePrice()
    instance_pricing.multi_price(inputs[1], inputs[2], inputs[0], inputs[3], inputs[4])
    instance_pricing.multipipeline(int(inputs[5]))
    traces = []
    
    count = 1
    for key, value in instance_pricing.prices.items():
        
#         if count == 9:
#             color1 = 'purple'
#         else:
#             color1 = 'orange'
        
        all_values = list(value)[-3:] + list(instance_pricing.forecasted_values[key])
  
        traces.append(go.Scatter(
      
        x = instance_pricing.date_range[-3:].append(instance_pricing.date_range[-int(inputs[5]):] + pd.DateOffset(int(inputs[5]))),
        y = all_values,
        mode='lines',
        connectgaps=True,
        name = key,
#         line = dict(
#         color = color1),

    ))
        count +=1
# name = key
    data = traces
# title = f'AWS EC2 {inputs[0]} Spot Price'
    layout = dict(title = f'AWS EC2 {inputs[0]} Spot Price' ,
              xaxis = dict(title = 'Month'),
              yaxis = dict(title = 'Price ($)'),
              )
    fig = dict(data=data, layout=layout)
    final_message = py.iplot(fig, filename='basic-line')
  

    return render_template('index.html', message=final_message)

if __name__ == '__main__':
    app.run()
