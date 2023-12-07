import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Chargement des données depuis un fichier CSV
data = pd.read_csv("clean_data.csv")

# Prétraitement des données
data = data.drop(['step', 'nameOrig', 'nameDest'], axis=1)
data = pd.get_dummies(data, columns=['type'], drop_first=True)

# Sélection de certaines colonnes pour la démonstration de K-Means (à adapter selon vos besoins)
selected_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
X = data[selected_columns]

# Normalisation des données (importante pour K-Means)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Application de l'algorithme K-Means
k = 2  # Nombre de clusters à former
kmeans = KMeans(n_clusters=k, random_state=42)
data['cluster'] = kmeans.fit_predict(X_scaled)

# Visualisation des clusters en fonction de deux caractéristiques (à adapter selon vos besoins)
feature1 = 'amount'
feature2 = 'oldbalanceOrg'

plt.scatter(data[data['cluster'] == 0][feature1], data[data['cluster'] == 0][feature2], label='Cluster 0')
plt.scatter(data[data['cluster'] == 1][feature1], data[data['cluster'] == 1][feature2], label='Cluster 1')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, c='red', label='Centroids')
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.title('K-Means Clustering')
plt.legend()
plt.show()


