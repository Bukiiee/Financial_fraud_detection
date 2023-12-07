import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Charger les données CSV
file_path = 'Financial_Fraud.csv'  # Remplacez par le chemin réel de votre fichier CSV
data = pd.read_csv(file_path)

# Encodage des colonnes catégorielles si nécessaire
label_encoder = LabelEncoder()
data['type'] = label_encoder.fit_transform(data['type'])
data['nameOrig'] = label_encoder.fit_transform(data['nameOrig'])
data['nameDest'] = label_encoder.fit_transform(data['nameDest'])

# Séparation des features et de la cible
features = ['step', 'type', 'amount', 'nameOrig', 'oldBalanceOrig', 'newBalanceOrig', 'nameDest', 'oldBalanceDest', 'newBalanceDest', 'isFlaggedFraud']  
target = 'isFraud'  
X = data[features]
y = data[target]

# Séparation des données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du modèle de régression Ridge
alpha = 1.0  # Paramètre de régularisation, vous pouvez ajuster selon vos besoins
model = Ridge(alpha=alpha)

# Entraînement du modèle
model.fit(X_train, y_train)

# Prédiction sur les données de test
y_pred = model.predict(X_test)

# Calcul de l'erreur moyenne quadratique
mse = mean_squared_error(y_test, y_pred)
print("Erreur moyenne quadratique :", mse)