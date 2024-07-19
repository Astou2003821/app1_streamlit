import pandas as pd
df = pd.read_csv('Expresso_churn_dataset.csv')
df.head()

df.info()

df.describe()

df.isnull().sum()

df.fillna(df.mode().iloc[0], inplace=True)

df.duplicated().sum()

from matplotlib import pyplot
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(20,15))

sns.boxplot(df)

plt.title('valeurs aberrantes')

plt.ylabel('Valeurs')

plt.show()

from sklearn.preprocessing import LabelEncoder

for col in df.columns:

    if df[col].dtype == 'object':

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col])

df

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

import matplotlib.pyplot as plt

X = df[["FREQUENCE", "REGULARITY"]]
y = df["CHURN"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000, random_state=42)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

y_pred_proba = clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)

classification_rep = classification_report(y_test, y_pred)

confusion_mat = confusion_matrix(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_pred_proba)


print(f'Accuracy: {accuracy}')

print('Classification Report:')

print(classification_rep)

print('Confusion Matrix:')

print(confusion_mat)

print(f'ROC AUC Score: {roc_auc}')

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('Taux de vrai positif')

plt.ylabel('Taux de vrai négatif')

plt.title('courbe ROC')

plt.legend(loc="lower right")

plt.show()


import joblib

joblib.dump(clf, 'logistic_regression_model.pkl')

joblib.dump(X.columns, 'model_columns.pkl')



import streamlit as st

import numpy as np

model = joblib.load('logistic_regression_model.pkl')

model_columns = joblib.load('model_columns.pkl')

st.title("Prédiction de Expresso churn")

st.write("""Cette application utilise un modèle de régression logistique pour prédire la probabilité de désabonnement des clients.
Entrez les caractéristiques du client pour obtenir une prédiction.""")

#caractéristiques du client
def get_user_input():

    inputs = {}

    for column in model_columns:

        inputs[column] = st.sidebar.number_input(f"{column}", value=0)

    return pd.DataFrame([inputs])

# Obtenir les caractéristiques de l'utilisateur
user_input = get_user_input()

#Ajout d'un bouton de validation
if st.sidebar.button("Valider"):

    # Afficher les caractéristiques de l'utilisateur
    st.subheader("Caractéristiques de l'utilisateur")

# Afficher les caractéristiques de l'utilisateur
st.subheader("Caractéristiques de l'utilisateur")

st.write(user_input)

# Faire la prédiction
prediction_proba = model.predict_proba(user_input)[:, 1][0]

prediction = model.predict(user_input)[0]

# Afficher la prédiction
st.subheader("Prédiction")
churn_status = "Oui" if prediction == 1 else "Non"

st.write(f"Le client va-t-il churn? {churn_status}")

# Afficher la probabilité
st.subheader("Probabilité de churn")

st.write(f"{prediction_proba:.2f}")

#Exécution de l'application Streamlit



