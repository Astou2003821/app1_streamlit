import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib

df = pd.read_csv('df.csv')

st.title("Prédiction de Expresso churn")

st.write("""Cette application utilise un modèle de régression logistique pour prédire la probabilité de désabonnement des clients.""")

st.write('## Dataframe avant nettoyage')
st.dataframe(df.head())

st.write('## Info du dataframe')
st.write(df.info())

st.write('## Valeurs manquantes du dataframe')
st.dataframe(df.isnull().sum())

st.write('## gerer les Valeurs manquantes')
df.fillna(df.mode().iloc[0], inplace=True)

st.write('## Valeurs manquantes après traitement')
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
y_pred_proba = clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

st.write(f'Accuracy: {accuracy}')
st.write('Classification Report:')
st.text(classification_rep)
st.write('Confusion Matrix:')
st.dataframe(confusion_mat)
st.write(f'ROC AUC Score: {roc_auc}')

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC')
plt.legend(loc="lower right")
st.pyplot(plt)

st.write('## Chargement du modèle') 
model = joblib.load('logistic_regression_model.pkl')
model_columns = joblib.load('model_columns.pkl')
st.subheader("Caractéristiques de l'utilisateur")

def get_user_input():
    inputs = {}
    for column in model_columns:
        inputs[column] = st.sidebar.number_input(f"{column}", value=0)
    return pd.DataFrame([inputs])

user_input = get_user_input()

if st.sidebar.button("Valider"):
    st.subheader("Caractéristiques de l'utilisateur")
    st.write(user_input)

    prediction_proba = model.predict_proba(user_input)[:, 1][0]
    prediction = model.predict(user_input)[0]

    st.write('## Prédiction')
    churn_status = "Oui" if prediction == 1 else "Non"
    st.write(f"Le client va-t-il churn? {churn_status}")

    st.write('## Probabilité de churn')
    st.write(f"{prediction_proba:.2f}")
