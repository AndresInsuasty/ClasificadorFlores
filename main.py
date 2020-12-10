# importar librerias
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble  import RandomForestClassifier

# Titulo

st.write("""
# Clasificando Flores Udenar
""")

st.sidebar.header('Ingrese los parametros para predecir')

def parametros():
    SL = st.sidebar.slider('Sepalo-L',4.3,7.9,5.4)
    SW = st.sidebar.slider('Sepalo-W',2.0,4.4,3.4)
    PL = st.sidebar.slider('Petalo-L',1.0,6.9,1.3)
    PW = st.sidebar.slider('Petalo-W',0.1,2.5,0.2)
    data = {
        'SL':SL,
        'SW':SW,
        'PL':PL,
        'PW':PW
    }

    predictores = pd.DataFrame(data,index=[0])
    return predictores

df = parametros()

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X,Y)

prediccion = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.write(prediccion)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediccion])

st.subheader('Prediction Probability')
st.write(prediction_proba)