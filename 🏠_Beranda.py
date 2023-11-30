import streamlit as st
import pandas as pd
from st_aggrid import AgGrid


# from sklearn import datasets
# import pandas as pd
# import numpy as np
# from sklearn import datasets
# from sklearn.decomposition import PCA
# import plotly.express as px

# iris = datasets.load_iris()
# X_reduced = PCA(n_components=3).fit_transform(iris.data)
# df_iris_pca = pd.DataFrame(X_reduced, columns=["1st Eigenvector", "2nd Eigenvector", "3rd Eigenvector"])
# df_iris_pca['target'] = iris.target
# df_iris_pca['target'] = df_iris_pca.target.replace({0:iris.target_names[0],
#                                                     1:iris.target_names[1],
#                                                     2:iris.target_names[2]})
# fig = px.scatter_3d(df_iris_pca, 
#                     x='1st Eigenvector', 
#                     y='2nd Eigenvector', 
#                     z='3rd Eigenvector',
#                     color='target')

st.set_page_config(
    page_title="Klasifikasi Bunga Iris",
    page_icon="logo_hoda.png",
)

st.write("# 🌷 :red[Klasifikasi Bunga Iris] 🌷")

# st.sidebar.success("Select a demo above.")

st.markdown(""":rainbow[--------------------------------------------------------------------------------------------------------------------------------------------]""")
st.markdown('<div style="text-align: center"> <b>وَلَا تَلْبِسُوا۟ ٱلْحَقَّ بِٱلْبَـٰطِلِ وَتَكْتُمُوا۟ ٱلْحَقَّ وَأَنتُمْ تَعْلَمُونَ</b> </div>',  unsafe_allow_html=True)
st.markdown('<div style="text-align: center , font-family:   Helvetica, sans-serif"> Dan janganlah kalian campuradukkan <b> kebenaran dengan kebatilan </b> dan (janganlah) kalian sembunyikan kebenaran, sedangkan kalian mengetahuinya.</div>',  unsafe_allow_html=True)
st.markdown('<div style="text-align: center"> <b>QS Al-Baqarah : 42</b> </div>',  unsafe_allow_html=True)
st.markdown(""":rainbow[--------------------------------------------------------------------------------------------------------------------------------------------]""")

st.markdown(
    """
    Klasifikasi Bunga Iris merupakan contoh implementasi Klasifikasi menggunakan
    model Machine Learning yang biasanya dipelajari pertama kali.
"""
)

st.markdown(
    """
    **👈 Anda bisa mengakses Web App Klasifikasi Bunga Iris** pada sidebar di samping!

    ### Dataset Bunga Iris
"""
)

st.image("Iris-Dataset-Classification.png")

st.markdown(
    """
    - Data Bunga Iris di ambil dari dataset scikit-learn berikut [Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)
    - Ini merupakan dataset ukuran panjang dan lebar dari petal dan sepal 3 jenis Bunga Iris (Setosa, Versicolour, dan Virginica).
    - Data ini disimpan dalam numpy array berukuran 150x4
    - Dataset ini memiliki 150 baris sampel dengan 4 kolom yaitu Petal Length, Petal Width, Sepal Length, Sepal Width
    - Hipotesis dari pemodelan klasifikasi dataset ini adalah : Kita dapat mengklasifikasikan Bunga Iris berdasarkan data ukuran Sepal dan Petal
    

    ### Visualisasi Dataset Bunga Iris

    Berikut adalah visualisasi dataset iris dengan yang telah di transformasi menggunakan Principal Component Analysis(PCA):

"""
)

st.image("plot_pca_iris.png")
# st.plotly_chart(fig, use_container_width=True)

st.markdown(
"""
    ### Pemodelan Klasifikasi Bunga Iris
    Berikut kami telah melakukan pemodelan klasifikasi data Bunga Iris dengan hasil sebagai berikut:
"""
)

model= ['Support Vector Machine', 'Naive Bayes', 'Decicion Tree', 'Gradien Boosting', 'K-Nearest Neighbor','Random Forest']
acc =[0.9777,0.9555,0.9555,0.9555,0.9777,0.9777]
pre = [0.9790,0.9555,0.9555,0.9555,0.9796,0.9790]
rec =[0.9777,0.9555,0.9555,0.9555,0.9777,0.9777]
f1s= [0.9776,0.9555,0.9555,0.9555,0.9779,0.9776]

df_pre= pd.DataFrame({ 'Model' : model,
                        'Accuracy' : acc,
                        'Precision':pre,
                        'Recall' : rec,
                        'F1-Score' : f1s
})

df_p = df_pre.style.highlight_max(color = 'red', axis = 0)
# df_p_f = AgGrid(df_pre)
st.dataframe(df_p, use_container_width=True, hide_index=True)

st.markdown(
"""
    Berdasarkan hasil di atas, maka akan diimplementasi model **:red[K-Nearest Neighbour]** pada web-app ini
"""
)
st.markdown(""":rainbow[--------------------------------------------------------------------------------------------------------------------------------------------]""")

st.markdown('<div style="text-align: center"> <b>Hobi Data © 2023</b> </div>',  unsafe_allow_html=True)
st.markdown('<div style="text-align: center"> Contributor : Joko Eliyanto, Indra Cahya Ramdani </div>', unsafe_allow_html=True)
    
