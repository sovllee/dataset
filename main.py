import streamlit as st
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

st.title('Streamlit example')

st.write("""
# Explore different classifier
which one is the best?
""")

dataset_name = st.sidebar.selectbox("Select dataset", ('Iris',"Breast Cancer","Wine dataset"))


classifier_name=st.sidebar.selectbox("Select classifier", ('KNN',"SVN","Random Forest"))

def get_dataset(dataset_name):
    if dataset_name=="Iris":
        data=datasets.load_iris()
    elif dataset_name=="Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X=data.data
    y=data.target
    return X,y

X,y=get_dataset(dataset_name)
st.write('shape of dataset',X.shape)
st.write('number of classes',len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name=='KNN':
        K=st.sidebar.slider("K",1,15)
        params["K"]=K
    elif clf_name=='SVN':
        C=st.sidebar.slider("C",0.01,10.0)
        params['C']=C
    else:
        max_depths=st.sidebar.slider("max_depth",2,15)
        n_estimators=st.sidebar.slider("n_estimators",1,100)
        params['max_depth']=max_depths
        params['n_estimators']=n_estimators

    return params
    
params=add_parameter_ui(classifier_name)

def get_classifier(clf_name,params):
    if clf_name=='KNN':
        clf=KNeighborsClassifier(n_neighbors=params['K'])
    elif clf_name=='SVN':
        clf=SVC(C=params["C"])
        
    else:
        clf=RandomForestClassifier(n_estimators=params["n_estimators"],
                                   max_depth=params['max_depth'],random_state=1234)
    return clf

clf=get_classifier(classifier_name,params)

#classification
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)

clf.fit(X_train,y_train)

y_predict=clf.predict(X_test)

acc=accuracy_score(y_test,y_predict)
st.write(f"classifier= {classifier_name}")
  
st.write(f'accuracy={acc}')

#PLOT
pca=PCA(2)

X_projected=pca.fit_transform(X)

x1 = X_projected[:,0]
x2 = X_projected[:,2]



