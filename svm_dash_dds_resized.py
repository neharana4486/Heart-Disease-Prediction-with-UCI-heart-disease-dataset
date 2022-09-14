# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 22:59:40 2020

@author: the Dash Girlz and Sleepy Hermit and Khadija and Neha
"""

import pandas as pd
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate  # this will be necessary so the page doesn't upload until everything is added


import pandas as pd
import numpy as np
data = pd.read_csv('cdatasetfilled.csv')
from sklearn import preprocessing
from sklearn import metrics
from sklearn import tree
from sklearn.svm import SVC



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv("cdatasetfilled.csv")
({
    'cp': list
}),

colors = {
    'background': '#DAF7A6',
    'text': '#090909'
}

#
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H4(
        children='MEDecide',
        style={
            'textAlign': 'center',
            'color': colors['text']

        }
    ),

    html.Div(children='a medical application to predict heart disease', style={
        'textAlign': 'center',
        'color': colors['text']
    }),

    html.H6(
        children='Age',),  # title of the Age cell
    dcc.Input(id="age", type="number", placeholder="Enter age"),  # age input
    html.Div(id="age-out"),

    html.H6('Sex'),
    dcc.Dropdown(
        id='sex-dropdown',
        options=[
            {'label': 'male', 'value': '1'},
            {'label': 'female', 'value': '0'},
        ],

        placeholder="Select...",
        searchable=True,
        clearable=True,
         style=dict(
                    width='40%')
    ),
    html.Div(id='dd-output-container4'),

    html.H6('Chest pain type'),
    dcc.Dropdown(
        id='cp-dropdown',
        #
        options=[
            {'label': 'typical angina', 'value': '1'},
            {'label': 'atypical angina', 'value': '2'},
            {'label': 'non-anginal pain', 'value': '3'},
            {'label': 'asymptomatic', 'value': '4'}
        ],
        value='chest pain type',
        placeholder="Select...",
        searchable=True,
        clearable=True,
        style=dict(
                    width='40%')
    ),

    html.Div(id='dd-output-container'),

    html.H6('Number of major vessels colored by flouroscopy'),
    dcc.Dropdown(
        id='ca-dropdown',
        options=[
            {'label': '0', 'value': '0'},
            {'label': '1', 'value': '1'},
            {'label': '2', 'value': '2'},
            {'label': '3', 'value': '3'}
        ],
        value='number of major vessels colored by flouroscopy',
        placeholder="Select...",
        searchable=True,
        clearable=True,
        style=dict(
                    width='40%')
    ),
    html.Div(id='dd-output-container2'),

    html.H6('The heart status as retrieved from Thallium test'),  # title of the third dropdown THAL
    dcc.Dropdown(
        id='thal-dropdown',
        options=[
            {'label': 'normal', 'value': '3'},
            {'label': 'fixed defect', 'value': '6'},
            {'label': 'reversable defect', 'value': '7'}
        ],

        placeholder="Select...",
        searchable=True,
        clearable=True,
        style=dict(
                    width='40%')
    ),
    html.Div(id='dd-output-container3'),

    html.H6('Maximum heart rate achieved'),  # title of the thalach cell
    dcc.Input(id="thalac", type="number", placeholder="Enter heart rate"),  # thalac input
    html.Div(id="thalac-out"),

    html.H6('ST depression induced by exercise relative to rest'),  # title of the oldpeak cell
    dcc.Input(id="oldpeak", type="number", placeholder="Enter value"),  # oldpeak input
    html.Div(id="oldpeak-out"),

    html.H6('Exercise induced angina'),


    dcc.Slider(
        id='my-beloved-slider',
        min=0,
        max=1,
        step=None,
        marks={
            0: 'not present',
            1: 'present',
        },

        value=1,
        dots=True,
    ),
    html.Div(id='slider-output-container'), 


    html.H6('Resting blood pressure (in mm Hg on admission to the hospital)'),  # title of the trestbps cell
    dcc.Input(id="trestbps", type="number", placeholder="Enter blood pressure measurement"),  # bps input
    html.Div(id="trestbps-out"),

    html.H6('Serum cholesterol in mg/dl'),  # title of the chol cell
    dcc.Input(id="chol", type="number", placeholder="Enter cholesterol value"),  # chol input
    html.Div(id="chol-out"),

    html.H6('Fasting blood sugar'),
    dcc.Dropdown(
        id='bloodsugar-dropdown',
        options=[
            {'label': 'above 120 mg/dl', 'value': '1'},
            {'label': 'under or equal to 120 mg/dl', 'value': '0'},
        ],

        placeholder="Select...",
        searchable=True,
        clearable=True,
        style=dict(
                    width='40%')
    ),
    html.Div(id='dd-output-container5'),

    html.H6('Resting electrocardiographic results'),
    dcc.Dropdown(
        id='restecg-dropdown',
        options=[
            {'label': 'normal', 'value': '0'},
            {'label': 'having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)',
             'value': '1'},
            {'label': 'showing probable or definite left ventricular hypertrophy by Estes criteria', 'value': '2'}
        ],
        value='restecg',
        placeholder="Select...",
        searchable=True,
        clearable=True,
        style=dict(
                    width='40%')
    ),
    html.Div(id='dd-output-container6'),

    html.H6('The slope of the peak exercise ST segment'),
    dcc.Dropdown(
        id='slope-dropdown',
        options=[
            {'label': 'upsloping', 'value': '1'},
            {'label': 'flat', 'value': '2'},
            {'label': 'downsloping', 'value': '3'}
        ],
        value='slope',
        placeholder="Select...",
        searchable=True,
        clearable=True,
        style=dict(
                    width='40%')
    ),
    html.Div(id='dd-output-container7'),


    html.Div([
        html.Br(),

        html.Button(children='Submit', id='submit-button-state', n_clicks=0),
        html.Div(id='output-state', children='Press to calculate'),
    ],
        style={'text-align': 'center'}),


     html.Div(children=' 0 - Healthy; 1 - Diagnosed with stage 1; 2 - Diagnosed with stage 2; 3 - Diagnosed with stage 3; 4 - Diagnosed with stage 4', style={
        'textAlign': 'center',
        'color': colors['text']
    }),
])


@app.callback(Output('output-state', 'children'),
              [Input('submit-button-state', 'n_clicks')],
              [State('age', 'value')],
              [State('sex-dropdown', 'value')],
              [State('cp-dropdown', 'value')],
              [State('ca-dropdown', 'value')],
              [State('thal-dropdown', 'value')],
              [State('thalac', 'value')],
              [State('oldpeak', 'value')],
              [State('my-beloved-slider', 'value')],
              [State('trestbps', 'value')],
              [State('chol', 'value')],
              [State('bloodsugar-dropdown', 'value')],
              [State('restecg-dropdown', 'value')],
              [State('slope-dropdown', 'value')])
def update_output(n_clicks, age, sex, cp, ca, thal, thalac, oldpeak, exang, trestbps, chol, fbs, restecg, slope ):


    if n_clicks > 0:
        data = pd.read_csv('cdatasetfilled.csv')
        data.head()
        dataattrib = pd.read_csv("cdatasetfilled.csv",
                                 usecols=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang",
                                          "oldpeak", "slope", "ca", "thal", "num"])
        dataattrib.head()
        dataclass = data.iloc[:, -1]
        data['num'] = data.num.astype(int)
        dataclass.head()
        datasetmap = dataattrib[
            ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca",
             "thal"]]
        datasetmap.head()
        x = datasetmap.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        datasetmapnorm = pd.DataFrame(x_scaled, columns=datasetmap.columns)
        classifier_svm_rbf = SVC(kernel='rbf', gamma='auto')
        classifier_svm_rbf.fit(datasetmapnorm, dataclass)
        info = pd.DataFrame(data=np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalac, exang, oldpeak, slope, ca, thal]]),
                            columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang",
                                     "oldpeak", "slope", "ca", "thal"])
        x = info.values
        scaled = min_max_scaler.fit(datasetmap)
        x_scaled = min_max_scaler.transform(x)
        info = pd.DataFrame(x_scaled, columns=info.columns)
        predicted_disease = classifier_svm_rbf.predict(info)
        return 'The Button has been pressed {} times, Heart disease prediction is {}'.format(n_clicks, predicted_disease)
    else:
        return 'Enter the missing values'


if __name__ == '__main__':
    app.run_server(debug=True)