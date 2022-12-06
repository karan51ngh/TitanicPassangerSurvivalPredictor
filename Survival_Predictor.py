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
import multiprocessing as mp

st.set_page_config(
    page_title="The Notebook",
    page_icon="images/icon.png",
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
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

st.sidebar.image('images/logo.png')
st.image('images/header.png')

with st.form("my_form"):
    social_status = st.selectbox(
        "What's your Social Status",
        ('Working Class: Manual Labor', 'Middle Class: Moderately decent living standard', 'Upper Class: The Top 1%'))
    Agee = st.slider(
        'How old are you?', 1, 90, 30)
    Sibb = st.slider(
        'Number of Siblings+Spouces?', 0, 10, 2)
    Parchh = st.slider(
        'Number of Parents+Children?', 0, 6, 2)
    faree = st.slider(
        'how much would you be paying for your ticket, depending on your class?', 5, 500, 7)
    sexx = st.selectbox(
        "What's your sex?",
        ('Male', 'Female'))
    embarkk = st.selectbox(
        "Which town would you like to embark from?",
        ('Cherbourg', 'Queenstown', 'Southampton'))

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

### Liner Processing

        # logreg = LogisticRegression()
        # svc_classifier = SVC()
        # dt_classifier = DecisionTreeClassifier()
        # knn_classifier = KNeighborsClassifier(5)
        # rf_classifier = RandomForestClassifier(
        #     n_estimators=1000, criterion='entropy', random_state=42)
        # logreg.fit(xtrain, ytrain)
        # svc_classifier.fit(xtrain, ytrain)
        # dt_classifier.fit(xtrain, ytrain)
        # knn_classifier.fit(xtrain, ytrain)
        # rf_classifier.fit(xtrain, ytrain)
### End of Linear Processing

### Parallel Processing
        

        # def Initiate(models,data,queue):
        #     for i in range(len(models)):
        #         p = mp.Process(target=Parallel_training,args=(models[i],[xtrain,ytrain],i,queue))
        #         p.start()
        #         print("\nDone,",i,"\n")

        # fit_val = mp.Queue()
        # x = mp.Process(target=Initiate,args=(models,[xtrain,ytrain],fit_val))
        # x.start()
        # x.join()
        # print("Successfully joined")


            

### End of Parellel Processing

        # logreg_ypred = logreg.predict(xtest)
        # svc_classifier_ypred = svc_classifier.predict(xtest)
        # dt_classifier_ypred = dt_classifier.predict(xtest)
        # knn_classifier_ypred = knn_classifier.predict(xtest)
        # rf_classifier_ypred = rf_classifier.predict(xtest)

        # logreg_acc = accuracy_score(ytest, logreg_ypred)
        # svc_classifier_acc = accuracy_score(ytest, svc_classifier_ypred)
        # dt_classifier_acc = accuracy_score(ytest, dt_classifier_ypred)
        # knn_classifier_acc = accuracy_score(ytest, knn_classifier_ypred)
        # rf_classifier_acc = accuracy_score(ytest, rf_classifier_ypred)
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

        tempQ = 0
        tempS = 0
        if embarkk == 'Cherbourg':
            pass
        elif embarkk == 'Queenstown':
            tempQ = 1
        elif embarkk == 'Southampton':
            tempS = 1

        yy = np.array([social_status, Agee, Sibb,
                      Parchh, faree, sexx, tempQ, tempS])



        models=[]
        logreg = LogisticRegression()
        models.append(logreg)
        svc_classifier = SVC()
        models.append(svc_classifier)
        dt_classifier = DecisionTreeClassifier()
        models.append(dt_classifier)
        knn_classifier = KNeighborsClassifier(5)
        models.append(knn_classifier)
        rf_classifier = RandomForestClassifier(
            n_estimators=1000, criterion='entropy', random_state=42)
        models.append(rf_classifier)
 
        def Parallel_training(model,data,queue):
            model.fit(data[0],data[1])
            res = model.predict(data[2])
            queue.put(res)

        
        queue = mp.Queue()
        x1 = mp.Process(target=Parallel_training,args=(logreg,[xtrain,ytrain,[yy]],queue))
        x2 = mp.Process(target=Parallel_training,args=(svc_classifier,[xtrain,ytrain,[yy]],queue))
        x3 = mp.Process(target=Parallel_training,args=(dt_classifier,[xtrain,ytrain,[yy]],queue))
        x4 = mp.Process(target=Parallel_training,args=(knn_classifier,[xtrain,ytrain,[yy]],queue))
        x5 = mp.Process(target=Parallel_training,args=(rf_classifier,[xtrain,ytrain,[yy]],queue))
        
        x1.start()
        x2.start()
        x3.start()
        x4.start()
        x5.start()

        x1.join()
        x2.join()
        x3.join()
        x4.join()
        x5.join()

        # tempLR = logreg.predict([yy])
        # tempSV = svc_classifier.predict([yy])
        # tempDT = dt_classifier.predict([yy])
        # tempKN = knn_classifier.predict([yy])
        # tempRF = rf_classifier.predict([yy])

        # def Parallel_testing(model,data,queue):
        #     res = data.predict(data)
        #     queue.put(res)

        # def Initiate_pre(fit_val,data,queue):
        #     while fit_val.qsize():
        #         item = fit_val.get()
        #         st.write(item)
        #         p = mp.Process(target=Parallel_testing,args=(item[0],data,queue))
        #         p.start()
        # result = mp.Queue()
        # x = mp.Process(target=Initiate_pre,args=(fit_val,[yy],result))
        # x.start()
        # x.join()
        # solarr=[]
        # print(result)
        # while result.qsize():
        #     print(result.get())
            



##End of Prallel Prediction
        solarr=[]
        while queue.qsize():
            solarr.append(queue.get())
        st.write(solarr)
        countt = 0
        for i in solarr:
            if i == 1:
                countt += 1

        if countt >= 3:
            st.write('Odds would be in you favor, you might survive')
            st.balloons()
        else:
            st.write(
                "You're lucky you didn't board the ship that day! Odds might not have been in your favor")
            st.snow()
