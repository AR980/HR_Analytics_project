import numpy as np
import pandas as pd
import streamlit as st
import pickle
import warnings
warnings.filterwarnings('ignore')

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

with st.form("my_form"):
       room_bed = st.number_input('Enter room_bed',format="%.2f")
       room_bath = st.number_input('Enter room_bath',format="%.2f")
       living_measure = st.number_input('Enter living_measure',format="%.2f")
       lot_measure = st.number_input('Enter lot_measure',format="%.2f")
       ceil =st.number_input('Enter ceil ',format="%.2f")
       coast = st.number_input('Enter coast',format="%.2f")
       sight = st.number_input('Enter sight',format="%.2f")
       condition = st.number_input('Enter condition',format="%.2f")
       quality = st.number_input('Enter quality',format="%.2f")
       ceil_measure = st.number_input('Enter ceil_measure',format="%.2f")
       basement = st.number_input('Enter basement',format="%.2f")
       yr_built = st.number_input('Enter yr_built',format="%.2f")
       living_measure15 = st.number_input('Enter living_measure15',format="%.2f")
       lot_measure15 = st.number_input('Enter lot_measure15',format="%.2f")
       furnished = st.number_input('Enter furnished',format="%.2f")
       total_area =st.number_input('Enter total_area',format="%.2f")
       has_basement = st.text_input("Enter has_basement")
       house_land_ratio = st.number_input('Enter house_land_ratio',format="%.2f")
       has_renovated = st.text_input("Enter has_renovated")

       df_new = pd.DataFrame([
       room_bed ,
       room_bath ,
       living_measure ,
       lot_measure ,
       ceil ,
       coast ,
       sight ,
       condition ,
       quality ,
       ceil_measure ,
       basement ,
       yr_built ,
       living_measure15 ,
       lot_measure15 ,
       furnished ,
       total_area ,
       has_basement ,
       house_land_ratio ,
       has_renovated]).T

       df_new.set_axis(df.columns, axis='columns', inplace=True)

       df = df.append(df_new, ignore_index = True)
       dff = pd.get_dummies(df, columns=['room_bed', 'room_bath', 'ceil', 'coast', 'sight', 'condition', 'quality', 'furnished', 
                                          'has_basement', 'has_renovated'],drop_first=True)

       dff=dff[selected_columns]
       submitted = st.form_submit_button("Submit")
       if submitted:
              st.write("Price is:", loaded_model.predict([dff.iloc[-1,:].values])[0])