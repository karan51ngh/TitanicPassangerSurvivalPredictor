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
    # layout="wide",
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


st.header("Importing and Inspecting the Data")


df = pd.read_csv('data/train.csv')
st.dataframe(df)
st.write("Our Raw data, as imported")
st.dataframe(df.describe())


st.header("Data Cleaning")

col1, col2, col3 = st.columns(3)
with col1:
    # st.write("Checking missing values")
    # st.dataframe(df.isna().sum())  # dropping column, coz many missing values
    pass

with col2:
    df.drop(columns=['Cabin'], inplace=True)
    df["Embarked"] = df["Embarked"].fillna(value='S')

    mean = df["Age"].mean()
    std = df["Age"].std()
    is_null = df["Age"].isnull().sum()

    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size=is_null)

    # fill NaN values in Age column with random values generated
    age_slice = df["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    df["Age"] = age_slice
    # st.write("Missing values fixed")
    # st.dataframe(df.isna().sum())

male = pd.get_dummies(df['Sex'])
df = pd.concat([df, male], axis=1)
embark = pd.get_dummies(data=df['Embarked'])
df = pd.concat([df, embark], axis=1)
df.drop(columns=['female', 'C'], inplace=True)

st.write("Our Data, after Cleaning")
st.dataframe(df)


st.header("Numerical Value Analysis")

col1, col2 = st.columns(2)
with col1:
    st.write("Heat map to inspect co-orelation between different values")
    fig, ax = plt.subplots()
    sns.heatmap(df[["Survived", "SibSp", "Parch",
                    "Age", "Fare", "male", "Q", "S"]].corr(), annot=True, ax=ax)
    st.pyplot(fig)

with col2:
    for i in range(15):
        st.write("")
    st.write(
        '''
        Conclusion :

Only Fare and Gender seem to have a significative correlation with the survival probability.

It doesn't mean that the other features are not usefull. Subpopulations in these features can be correlated with the survival.
To determine this, we need to explore in detail these features
        ''')

col1, col2 = st.columns(2)

with col1:
    st.write("Number of siblings per each person")
    bargraph_sibsp = sns.factorplot(
        x="SibSp", y="Survived", data=df, kind="bar", size=8)
    st.pyplot(bargraph_sibsp)


with col2:
    for i in range(15):
        st.write("")
    st.write(
        '''

        It seems that passengers having a lot of siblings/spouses have less chance to survive.
     	Single passengers(0 SibSP) or with two other persons(SibSP 1 or 2) have more chance to survive.
        ''')


# AGE

col1, col2 = st.columns(2)
with col1:
    age_visual = sns.FacetGrid(df, col='Survived', size=7)
    age_visual = age_visual.map(sns.distplot, "Age")
    age_visual = age_visual.set_ylabels("survival probability")
    st.pyplot(age_visual)

with col2:
    for i in range(7):
        st.write("")
    st.write(
        '''
	Age distribution seems to be a tailed distribution, maybe a gaussian distribution.

	We notice that age distributions are not the same in the survived and not survived subpopulations. Indeed, there is a peak corresponding to young passengers, that have survived. We also see that passengers between 60-80 have less survived.

	So, even if "Age" is not correlated with "Survived", we can see that there is age categories of passengers that of have more or less chance to survive.

	It seems that very young passengers have more chance to survive.
	'''
    )

# SEX

# plt.figure(figsize=(12, 10))
# plt.figure(figsize=(12, 10))
col1, col2 = st.columns(2)
with col1:
    fig = plt.figure()
    age_plot = sns.barplot(x="Sex", y="Survived", data=df)
    age_plot = age_plot.set_ylabel("Survival Probability")
    st.pyplot(fig)
with col2:
    for i in range(7):
        st.write("")
    st.dataframe(df[["Sex", "Survived"]].groupby('Sex').mean())
    for i in range(4):
        st.write("")
    st.write('''
    It is clearly obvious that Male have less chance to survive than Female. So Sex, might play an important role in the prediction of the survival.
          ''')
# PClass
col1, col2 = st.columns(2)
with col1:
    pclass = sns.factorplot(x="Pclass", y="Survived",
                            data=df, kind="bar", size=8)
    pclass = pclass.set_ylabels("survival probability")
    st.pyplot(pclass)

with col2:
    for i in range(15):
        st.write("")
    st.write('''
    It is clearly obvious that Lower Class members have less chance to survive than the higher class. 
    So Pclass, might play an important role in the prediction of the survival.
          ''')

# pclass vs survived sex
col1, col2 = st.columns(2)
with col1:
    g = sns.factorplot(x="Pclass", y="Survived", hue="Sex",
                       data=df, size=6, kind="bar")
    g = g.set_ylabels("survival probability")
    st.pyplot(g)

# Embarked
col1, col2 = st.columns(2)
with col2:
    st.dataframe(df["Embarked"].value_counts())
with col1:
    g = sns.factorplot(x="Embarked", y="Survived", data=df, size=7, kind="bar")
    g = g.set_ylabels("survival probability")
    st.pyplot(g)


g = sns.factorplot("Pclass", col="Embarked",  data=df, size=7, kind="count")
g.despine(left=True)
g = g.set_ylabels("Count")
st.pyplot(g)

g = sns.factorplot("Sex", col="Embarked",  data=df, size=7, kind="count")
st.pyplot(g)


# preparing data for training:

x = df.drop(columns=['Survived', 'Name', 'PassengerId',
            'Sex', 'Embarked', 'Ticket'], axis=1)
y = df['Survived']

xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, test_size=0.30, random_state=0)

sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)

# classification

logreg = LogisticRegression()
svc_classifier = SVC()
dt_classifier = DecisionTreeClassifier()
knn_classifier = KNeighborsClassifier(5)
rf_classifier = RandomForestClassifier(
    n_estimators=1000, criterion='entropy', random_state=42)

logreg.fit(xtrain, ytrain)
svc_classifier.fit(xtrain, ytrain)
dt_classifier.fit(xtrain, ytrain)
knn_classifier.fit(xtrain, ytrain)
rf_classifier.fit(xtrain, ytrain)

logreg_ypred = logreg.predict(xtest)
svc_classifier_ypred = svc_classifier.predict(xtest)
dt_classifier_ypred = dt_classifier.predict(xtest)
knn_classifier_ypred = knn_classifier.predict(xtest)
rf_classifier_ypred = rf_classifier.predict(xtest)

# finding accuracy

logreg_acc = accuracy_score(ytest, logreg_ypred)
svc_classifier_acc = accuracy_score(ytest, svc_classifier_ypred)
dt_classifier_acc = accuracy_score(ytest, dt_classifier_ypred)
knn_classifier_acc = accuracy_score(ytest, knn_classifier_ypred)
rf_classifier_acc = accuracy_score(ytest, rf_classifier_ypred)

st.write("Logistic Regression : ", round(logreg_acc*100, 2))
st.write("Support Vector      : ", round(svc_classifier_acc*100, 2))
st.write("Decision Tree       : ", round(dt_classifier_acc*100, 2))
st.write("K-NN Classifier     : ", round(knn_classifier_acc*100, 2))
st.write("Random Forest       : ", round(rf_classifier_acc*100, 2))
