# import pandas as dp
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
#
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler,normalize
# from sklearn.metrics import silhouette_score
#
# data=pd.read_csv("health_data.csv")
# data.head()
# selected_cols=["age","sex","PhysHlth","DiffWalk"]
# normalized_df=(data-data.mean())/data.std()
# cluster_data=data.loc[:,selected_cols]
# cluster_range=[2,3,4,5,6,7,8,9,10,11,12,13,14]
# inertias=[]
# for c in cluster_range :
#     kmeans=KMeans(init='K-means++',n_clusters=c,n_init=100,random_state=0).fit(normalized_df)
#     inertias.append(kmeans.inertia_)
#
# plt.figure()
# plt.plot(cluster_range,inertias,marker='0')
# plt.show()
#
#
# from sklearn.metrics import silhouette_score, silhouette_samples
# clusters_range = range(2,15)
# results =[]
# for c in clusters_range:
#     clusterer = KMeans (init='k-means++',n_clusters=c,n_init=100,random_state=0)
#     clusterLabels = clusterer.fit_predict(normalized_df)
#     silhouette_avg=silhouette_score(normalized_df,clusterLabels)
#     results.append ([c,silhouette_avg])
# result=pd.DataFram(results,columns=['n_clusters','silhouette_score'])
# pivot_km = pd.pivottable(result, lndex="n_clusters", Values="silhouette_score")
# plt.figure()
# sns.heatmap(pivot_km, annot=True,linewidths=.5, fmt='.3f',cmap=sns.cm.rocket_r)
#
#
# kmeans_sel=KMeans (init='k-means++',n_clusters=3,n_init=100,random_state=1).fit(normalized_df)
# lables=pd.DataFrame(kmeans_sel.lables_)
# scatters=silhouette_samples.scatter(cluster_data,h='Cluster')
#
#
# grouped_km=cluster_data.groupby(['Cluster']).mean().round(1)
# print(grouped_km)
#
# import numpy as nm
# import matplotlib.pyplot as mtp
# import pandas as pd
# dataset = pd.read_csv('health_data.csv')
# x = dataset.iloc[:, [3, 4]].values
#
# # finding optimal number of clusters using the elbow method
# from sklearn.cluster import KMeans
#
# wcss_list = []  # Initializing the list for the values of WCSS
#
# # Using for loop for iterations from 1 to 18.
# for i in range(1, 18):
#     kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
#     kmeans.fit(x)
#     wcss_list.append(kmeans.inertia_)
# mtp.plot(range(1, 18), wcss_list)
# mtp.title('The Elobw Method Graph')
# mtp.xlabel('Number of clusters(k)')
# mtp.ylabel('wcss_list')
# mtp.show()
#
# #training the K-means model on a dataset
# kmeans = KMeans(n_clusters=5, init='k-means++', random_state= 42)
# y_predict= kmeans.fit_predict(x)
# #visulaizing the clusters
# mtp.scatter(x[y_predict == 0, 0], x[y_predict == 0, 1], s = 100, c = 'blue', label = 'Cluster 1') #for first cluster
# mtp.scatter(x[y_predict == 1, 0], x[y_predict == 1, 1], s = 100, c = 'green', label = 'Cluster 2') #for second cluster
# mtp.scatter(x[y_predict== 2, 0], x[y_predict == 2, 1], s = 100, c = 'red', label = 'Cluster 3') #for third cluster
# mtp.scatter(x[y_predict == 3, 0], x[y_predict == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4') #for fourth cluster
# mtp.scatter(x[y_predict == 4, 0], x[y_predict == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5') #for fifth cluster
# mtp.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroid')
# mtp.title('Clusters of health data')
# mtp.xlabel('Annual Income (k$)')
# mtp.ylabel('Score (1-100)')
# mtp.legend()
# mtp.show()
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

db=pd.read_csv('health_data.csv')
x = db.Age.values.tolist()
y = db.BMI.values.tolist()
plt.scatter(x, y)
plt.show()

data = list(zip(x, y))
inertias = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)
plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
kmeans = KMeans(n_clusters=5)
kmeans.fit(data)
plt.scatter(x, y, c=kmeans.labels_)
plt.show()