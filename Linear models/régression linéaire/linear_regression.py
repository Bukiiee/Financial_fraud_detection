import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données CSV
data = pd.read_csv('Financial_Fraud.csv')

# Encodage des colonnes catégorielles si nécessaire
label_encoder = LabelEncoder()
data['type'] = label_encoder.fit_transform(data['type'])
data['nameOrig'] = label_encoder.fit_transform(data['nameOrig'])
data['nameDest'] = label_encoder.fit_transform(data['nameDest'])

# Séparation des features et de la cible
X = data.drop('isFraud', axis=1)
y = data['isFraud']

# Séparation des données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du modèle de régression linéaire
model = LinearRegression()

# Entraînement du modèle
model.fit(X_train, y_train)

# Prédiction sur les données de test
y_pred = model.predict(X_test)

# Calcul de l'erreur moyenne quadratique
mse = mean_squared_error(y_test, y_pred)
print("Erreur moyenne quadratique :", mse)

# Affichage du graphique de dispersion avec la ligne de régression
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Vraies étiquettes')
plt.ylabel('Prédictions')
plt.title('Graphique de Dispersion avec Régression Linéaire')
plt.show()
