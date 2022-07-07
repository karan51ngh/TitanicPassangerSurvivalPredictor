from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

st.set_page_config(
    page_title="The Notebook",
    # page_icon="pics/icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)
hide_menu_style = """
        <style>
        # MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

with st.form("my_form"):
    social_status = st.selectbox(
        "What's your Social Status",
        ('Working Class: Manual Labor', 'Middle Class: Moderately decent living standard', 'Upper Class: The Top 1%'))
    Agee = st.number_input('Insert Age')
    Sibb = st.number_input('Number of Siblings/Spouces')
    Parchh = st.number_input('Number of parents/children')
    faree = st.slider(
        'how much would you be paying for your ticket, depending on your class?', 5, 500, 5)
    sexx = st.selectbox(
        "What's your sex?",
        ('Male', 'Female'))

    submitted = st.form_submit_button("Predict!")
    if submitted:
        df = pd.read_csv('data/train.csv')
        df.drop(columns=['Cabin'], inplace=True)
        df["Embarked"] = df["Embarked"].fillna(value='S')
        mean = df["Age"].mean()
        std = df["Age"].std()
        is_null = df["Age"].isnull().sum()
        rand_age = np.random.randint(mean - std, mean + std, size=is_null)
        age_slice = df["Age"].copy()
        age_slice[np.isnan(age_slice)] = rand_age
        df["Age"] = age_slice
        male = pd.get_dummies(df['Sex'])
        df = pd.concat([df, male], axis=1)
        embark = pd.get_dummies(data=df['Embarked'])
        df = pd.concat([df, embark], axis=1)
        df.drop(columns=['female', 'C'], inplace=True)
        x = df.drop(columns=['Survived', 'Name', 'PassengerId',
                             'Sex', 'Embarked', 'Ticket'], axis=1)
        y = df['Survived']
        xtrain, xtest, ytrain, ytest = train_test_split(
            x, y, test_size=0.30, random_state=0)
        xtrain = xtrain.to_numpy()
        xtest = xtest.to_numpy()
        logreg = LogisticRegression()
        rf_classifier = RandomForestClassifier(
            n_estimators=1000, criterion='entropy', random_state=42)
        logreg.fit(xtrain, ytrain)
        rf_classifier.fit(xtrain, ytrain)
        logreg_ypred = logreg.predict(xtest)
        rf_classifier_ypred = rf_classifier.predict(xtest)
        logreg_acc = accuracy_score(ytest, logreg_ypred)
        rf_classifier_acc = accuracy_score(ytest, rf_classifier_ypred)
        # st.write(logreg_acc)
        # st.write(rf_classifier_acc)
        if social_status == 'Working Class: Manual Labor':
            social_status = 3
        elif social_status == 'Middle Class: Moderately decent living standard':
            social_status = 2
        elif social_status == 'Upper Class: The Top 1%':
            social_status = 1

        Agee = int(Agee)
        Sibb = int(Sibb)
        Parchh = int(Parchh)
        faree = int(faree)
        if sexx == 'Male':
            sexx = 1
        elif sexx == 'Female':
            sexx = 0

        yy = np.array([social_status, Agee, Sibb, Parchh, faree, sexx, 0, 1])

        temp = logreg.predict([yy])

        if temp == 1:
            st.write('Odds would be in you favor, you might survive')
        if temp == 0:
            st.write(
                "You're lucky you didn't board the ship that day! Odds might not have been in your favor")
