import streamlit as st
import time
import numpy as np
import pandas as pd

st.set_page_config(page_title="Web App Iris", page_icon="logo_hoda.png")

st.write("# 🌷 :red[Klasifikasi Bunga Iris] 🌷")
st.markdown(""":rainbow[--------------------------------------------------------------------------------------------------------------------------------------------]""")

c1,c2 = st.columns([2,3])

with c1:
    st.image("iris_.png")

with c2:
    petal_p = st.number_input('Panjang Petal')
    petal_l = st.number_input('Lebar Petal')
    sepal_p = st.number_input('Panjang Sepal')
    sepal_l = st.number_input('Lebar Sepal')

    prediksi = st.button("Prediksi")
st.markdown(""":rainbow[--------------------------------------------------------------------------------------------------------------------------------------------]""")

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X_train, X_test, ytrain, ytest = train_test_split(iris.data, iris.target, test_size=0.3, random_state=1234)

model_knn = KNeighborsClassifier(n_neighbors=3)

model_knn.fit(X_train,ytrain)

data_input = np.array([[sepal_l, sepal_p, petal_l, petal_p]])

if prediksi:
    predict_knn = model_knn.predict(data_input)
    predict_proba_knn = model_knn.predict_proba(data_input)
    if predict_knn == 0:
        st.image('0_setosa.png')
        st.dataframe(pd.DataFrame(predict_proba_knn, columns=iris.target_names).style.highlight_max(color = 'red', axis = 1), 
                    use_container_width=True,
                    hide_index=True)
        st.write("Berdasarkan data yang diinputkan, hasil klasifikasinya adalah **:red[Iris Setosa]**")
    elif predict_knn == 1:
        st.image('1_versicolor.png')
        st.dataframe(pd.DataFrame(predict_proba_knn, columns=iris.target_names).style.highlight_max(color = 'red', axis = 1), 
                    use_container_width=True,
                    hide_index=True)
        st.write("Berdasarkan data yang diinputkan, hasil klasifikasinya adalah **:red[Iris Versicolor]**")
    elif predict_knn == 2:
        st.image('2_virginica.png')
        st.dataframe(pd.DataFrame(predict_proba_knn, columns=iris.target_names).style.highlight_max(color = 'red', axis = 1), 
                use_container_width=True,
                hide_index=True)
        st.write("Berdasarkan data yang diinputkan, hasil klasifikasinya adalah **:red[Iris Virginica]**")

st.markdown(""":rainbow[--------------------------------------------------------------------------------------------------------------------------------------------]""")

st.markdown('<div style="text-align: center"> <b>Hobi Data © 2023</b> </div>',  unsafe_allow_html=True)
st.markdown('<div style="text-align: center"> Contributor : Joko Eliyanto, Indra Cahya Ramdani </div>', unsafe_allow_html=True)