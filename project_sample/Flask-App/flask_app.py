import numpy as np
import pandas as pd
import streamlit as st
import pickle
import warnings
warnings.filterwarnings('ignore')
from flask import Flask, request, render_template

app = Flask("__name__")

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

df=pd.read_csv('innercity_new.csv')
df=df.drop(['dayhours','month/year','price_bins', 'price'],axis=1)

selected_columns = ['living_measure', 'lot_measure', 'ceil_measure', 'yr_built',
       'living_measure15', 'lot_measure15', 'total_area', 'house_land_ratio',
       'room_bed_1.0', 'room_bed_2.0', 'room_bed_3.0', 'room_bath_0.5',
       'room_bath_0.75', 'room_bath_1.0', 'room_bath_1.25', 'room_bath_1.5',
       'room_bath_1.75', 'room_bath_2.0', 'room_bath_2.25', 'room_bath_2.5',
       'room_bath_2.75', 'room_bath_3.0', 'room_bath_3.25', 'room_bath_3.5',
       'room_bath_3.75', 'room_bath_4.0', 'room_bath_4.25', 'room_bath_4.5',
       'room_bath_4.75', 'room_bath_5.0', 'room_bath_5.25', 'ceil_1.5',
       'ceil_2.5', 'ceil_3.0', 'ceil_3.5', 'coast_1.0', 'sight_1.0',
       'sight_2.0', 'sight_3.0', 'sight_4.0', 'condition_2.0', 'condition_3.0',
       'condition_4.0', 'condition_5.0', 'quality_4.0', 'quality_5.0',
       'quality_6.0', 'quality_7.0', 'quality_8.0', 'quality_9.0',
       'quality_10.0', 'quality_11.0', 'quality_12.0', 'has_basement_Yes',
       'has_renovated_Yes']

@app.route("/")
def loadPage():
	return render_template('home.html', query="")


@app.route("/", methods=['POST'])
def predict():
    
    df=pd.read_csv('innercity_new.csv')
    df=df.drop(['dayhours','month/year','price_bins', 'price'],axis=1)

    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']
    inputQuery6 = request.form['query6']
    inputQuery7 = request.form['query7']
    inputQuery8 = request.form['query8']
    inputQuery9 = request.form['query9']
    inputQuery10 = request.form['query10']
    inputQuery11 = request.form['query11']
    inputQuery12 = request.form['query12']
    inputQuery13 = request.form['query13']
    inputQuery14 = request.form['query14']
    inputQuery15 = request.form['query15']
    inputQuery16 = request.form['query16']
    inputQuery17 = request.form['query17']
    inputQuery18 = request.form['query18']
    inputQuery19 = request.form['query19']

    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7, 
                inputQuery8, inputQuery9, inputQuery10, inputQuery11, inputQuery12, inputQuery13, inputQuery14,
                inputQuery15, inputQuery16, inputQuery17, inputQuery18, inputQuery19]]
        
    df_new = pd.DataFrame(data)
    print("yoyoy", len(df.columns), len(df_new.columns))
    df_new.set_axis(df.columns, axis='columns', inplace=True)
    
    df = df.append(df_new, ignore_index = True)
    dff = pd.get_dummies(df, columns=['room_bed', 'room_bath', 'ceil', 'coast', 'sight', 'condition', 'quality', 'furnished', 
                                            'has_basement', 'has_renovated'],drop_first=True)

    dff=dff[selected_columns]
    print(len(dff.columns))
    o1= loaded_model.predict([dff.iloc[-1,:].values])[0]
    
    return render_template('home.html', output1=o1, 
                           query1 = request.form['query1'], 
                           query2 = request.form['query2'],
                           query3 = request.form['query3'],
                           query4 = request.form['query4'],
                           query5 = request.form['query5'], 
                           query6 = request.form['query6'], 
                           query7 = request.form['query7'], 
                           query8 = request.form['query8'], 
                           query9 = request.form['query9'], 
                           query10 = request.form['query10'], 
                           query11 = request.form['query11'], 
                           query12 = request.form['query12'], 
                           query13 = request.form['query13'], 
                           query14 = request.form['query14'], 
                           query15 = request.form['query15'], 
                           query16 = request.form['query16'], 
                           query17 = request.form['query17'],
                           query18 = request.form['query18'], 
                           query19 = request.form['query19'])
    
app.run(debug=True)