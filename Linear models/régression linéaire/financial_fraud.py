import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Chargement des données depuis un fichier CSV
data = pd.read_csv("clean_data.csv")

# Prétraitement des données
data = data.drop(['step', 'nameOrig', 'nameDest'], axis=1)
data = pd.get_dummies(data, columns=['type'], drop_first=True)

# Séparation de 'isFraud' des autres colonnes
X = data.drop('isFraud', axis=1)
y = data['isFraud']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entraînement du modèle de régression logistique
logreg_model = LogisticRegression()
logreg_model.fit(X_train_scaled, y_train)

# Prédictions sur l'ensemble de test
y_pred = logreg_model.predict(X_test_scaled)

# Évaluation de la performance du modèle
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Afficher les résultats
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{classification_rep}")
