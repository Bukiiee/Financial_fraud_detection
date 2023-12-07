import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Chargement des données du dataset
data = pd.read_csv("./financial_fraud.csv")

# Pré-traitement des données
data = data.drop(['step', 'nameOrig', 'nameDest'], axis=1)

# Ajout d'une colonne isFlaggedFraud dans le DataFrame new_input
new_input = pd.DataFrame({
    'amount': [4892193.09],
    'oldbalanceOrg': [4892193.09],
    'newbalanceOrig': [4892193.09],
    'oldbalanceDest': [0.0],
    'newbalanceDest': [0.0],
    'type_CASH_OUT': [0],
    'type_DEBIT': [0],
    'type_PAYMENT': [0],
    'type_TRANSFER': [1],
    'isFlaggedFraud': [0]  # Placeholder for isFlaggedFraud
})

# Transformation en INT
data = pd.get_dummies(data, columns=['type'], drop_first=True)
# Ajout de la colonne isFlaggedFraud manquante
data['isFlaggedFraud'] = 0

# Séparation du 'isFraud' des autres colonnes
X = data.drop('isFraud', axis=1)
y = data['isFraud']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entraînement du modèle KNN
k = 5
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train_scaled, y_train)

# Normalisation de la nouvelle entrée
new_input_scaled = scaler.transform(new_input)

# Prédiction pour la nouvelle entrée
new_output = knn_model.predict(new_input_scaled)

print(f"Predicted Output for the New Input: {new_output}")
