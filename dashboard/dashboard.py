import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px

factor_df = pd.read_csv('https://drive.google.com/uc?id=1SzFkHl_wum8svcryhJHgLrDXKeGjg9CT')

def kmeans_sklearn(data, k):
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(data)
    silhouette_avg = silhouette_score(data, cluster_labels)
    return cluster_labels, silhouette_avg

# Finding optimal k from higgest Silhouette Score for model
best_silhouette_score = 0
best_k = None
k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]

for k in k_values:
    cluster_labels, silhouette_now = kmeans_sklearn(factor_df[['FAC1_1', 'FAC2_1']], k)
    if silhouette_now > best_silhouette_score:
        best_silhouette_score = silhouette_now
        best_k = k

kmeans = KMeans(n_clusters=best_k)
kmeans.fit(factor_df[['FAC1_1', 'FAC2_1']])
labels_sample = kmeans.labels_
centroids = kmeans.cluster_centers_
factor_df['label'] = labels_sample

# Plotly scatter plot
fig = px.scatter(
    factor_df,
    x='FAC1_1',
    y='FAC2_1',
    color='label',
    hover_data=['Provinsi'],
    labels={'FAC1_1': 'Bencana Hidrometeorologi', 'FAC2_1': 'Bencana Geologi'},
    title='Clustering Results with Factor Analysis'
)

# Adding centroids
centroids_df = pd.DataFrame(centroids, columns=['FAC1_1', 'FAC2_1'])
fig.add_trace(
    px.scatter(
        centroids_df,
        x='FAC1_1',
        y='FAC2_1'
    ).data[0]
)
fig.data[-1].marker.update(size=10, color='red', symbol='x')

fig.update_layout(
    plot_bgcolor='white',  # Change the background color
    paper_bgcolor='white',  # Change the outer paper color
    font_color='black',     # Change font color
    title_font_color='black',
    xaxis=dict(showgrid=True, gridcolor='gray'),
    yaxis=dict(showgrid=True, gridcolor='gray')
)

# Displaying plot in Streamlit
st.plotly_chart(fig)

fig.show()