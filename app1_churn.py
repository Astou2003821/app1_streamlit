import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow import keras
import model_loads as model

df = pd.read_csv('df.csv')

st.title("Prédiction de Expresso churn")

st.write("""Cette application utilise un modèle de régression logistique pour prédire la probabilité de désabonnement des clients.""")

st.write('## Dataframe avant nettoyage')
st.dataframe(df.head())

st.write('## Info du dataframe')
st.write(df.info())

st.write('## Valeurs manquantes du dataframe')
st.dataframe(df.isnull().sum())

st.write('## Valeurs aberrantes avec le boxplot')
plt.figure(figsize=(20, 15))
sns.boxplot(data=df)
plt.title('Valeurs aberrantes')
plt.ylabel('Valeurs')
st.pyplot(plt)

st.write('## Encodage du dataframe')
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

st.dataframe(df.head())

st.write('## Dataframe après nettoyage')
st.dataframe(df.head())

st.write('## Classifieur machine learning')

X = df[["FREQUENCE", "REGULARITY"]]
y = df["CHURN"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000, random_state=42)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

st.write(f'Accuracy: {accuracy}')
st.write('Classification Report:')
st.write(classification_rep)
st.write('Confusion Matrix:')
st.write(confusion_mat)
st.write(f'ROC AUC Score: {roc_auc}')

fpr, tpr, _ = roc_curve(y_test, y_pred)
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

fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot(color='gray', lw=2, linestyle='--')
plt.xlim()
plt.ylim()
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC')
plt.legend(loc="lower right")
st.pyplot(plt)

st.write('## Charger le modèle Keras')

def load_model():
    model = keras.models.load_model('logistic_regression_model.h5')
    return model

model = load_model()

st.write('## Titre de l-application')

st.title('Application de Régression avec Keras et Streamlit')

st.write('## Entrée utilisateur')
st.header('Entrer les caractéristiques du modèle')
input_1 = st.number_input('Entrée 1', value=0.0)
input_2 = st.number_input('Entrée 2', value=0.0)
input_3 = st.number_input('Entrée 3', value=0.0)

st.write('## Convertir les entrées en numpy array pour la prédiction')
input_data = np.array([[input_1, input_2, input_3]])

st.write('## Bouton de prédiction')
if st.button('Prédire'):
    prediction = model.predict(input_data)
    st.write(f'La prédiction est: {prediction[0][0]}')
    predictions = model.predict(input_data)
    st.write("Prédictions : ", predictions)
