import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn import metrics
from yellowbrick.classifier import ROCAUC
from mlxtend.plotting import plot_confusion_matrix


#read file
df = pd.read_csv("health_data.csv")
pd.set_option('display.max_columns',30)
print(df)
###
print(df.info())
print('Missing Values')
print(df.isnull().sum())
#Select unique values to descriptive statistics in the data set
dfa = pd.DataFrame(df)
print('HHAA')
print(dfa.describe())
unique_vals=[]
for col in df.columns:
    unival=df[col].nunique()
    unique_vals.append(unival)
#Presenting the findings using a dataframe
print(pd.DataFrame(unique_vals,columns=['Unique_Values'],index=df.columns))
print(df.columns)
#Index(['Age', 'Sex', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
#      'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
#      'HvyAlcoholConsump', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk',
#      'Diabetes', 'Hypertension', 'Stroke'],
#       dtype='object')

#






print("Presenting the countplots for categorical features")
for i in df.columns:
  fig, ax = plt.subplots(1,1, figsize=(15, 6))
  sns.countplot(y = df[i],data=df, order=df[i].value_counts().index, palette='flare')
  plt.ylabel(i)
  plt.yticks(fontsize=13)
  print(f'------------------------------{i}------------------------------')
  plt.box(False)
  plt.show()




fig=plt.figure()
#Create one or more subplots using add_subplot, because you can't create blank figure
ax = fig.add_subplot(1,1,1)
#Variable
ax.hist(df['Age'],bins = 5)
#Labels and Tit
plt.title('Age distribution')
plt.xlabel('Age')
plt.ylabel('#Number of children')
plt.show()
#
#
ax = fig.add_subplot(10,10,10)
#Variable
ax.scatter(df['PhysHlth'],df['GenHlth'])
#Labels and Tit
plt.title('PhysHlth and GenHlth distribution')
plt.xlabel('PhysHlth')
plt.ylabel('GenHlth')
plt.show()



plt.figure(figsize=(12,12),dpi=150)
sns.heatmap(df.corr(method='spearman'),vmin=0,fmt='.1f',annot=True,cmap='flare')
data=df.drop(['Sex','CholCheck','Smoker','Fruits','Veggies','HvyAlcoholConsump','MentHlth'],axis=1)
print(data.head())
#Splitting the data into input data features and target
X=data.drop('Stroke',axis=1)
y=data['Stroke']

#Splitting the data into input data features and target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=105,stratify=y)
print(X_train.shape)
print(X_test.shape)

#Scaling the data so that comparatively larger values do not make the model biased
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
accuracy_model=pd.DataFrame(columns=['Model','Accuracy'])


def model_train_test(model):
    model.fit(X_train_scaled,y_train)
    y_pred=model.predict(X_test_scaled)
    print(classification_report(y_test,y_pred))
    cm = metrics.confusion_matrix(y_test,y_pred)
    fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(8, 8),cmap='flare')
    plt.xlabel('Predictions')
    plt.ylabel('Actuals')
    plt.title('Confusion Matrix (Base Model)')
    plt.show()
    print('ROC-AUC\n')
    visualizer = ROCAUC(model, classes=["No", "Yes"])

    visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    visualizer.show()

#Testing Different Models
grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10)
model_train_test(logreg_cv)

#GAUSSIAN NAIVE BAYES
gnb=GaussianNB()
model_train_test(gnb)

#BERNOULLI NAIVE BAYES
bnb=BernoulliNB()
model_train_test(bnb)

#K-NEAREST NEIGHBOURS
knn = KNeighborsClassifier()

k_range = list(range(1, 20))
param_grid = dict(n_neighbors=k_range)

grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False, verbose=1)
grid_search = grid.fit(X_train_scaled, y_train)
#Fitting 10 folds for each of 19 candidates, totalling 190 fits
print(grid_search.best_params_)
accuracy = grid_search.best_score_ *100
print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy) )
#{'n_neighbors': 18}
#Accuracy for our training dataset with tuning is : 93.74%
model_train_test(KNeighborsClassifier(n_neighbors=18))

#DECISION TREE
param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
              'ccp_alpha': [0.1, .01, .001],
              'max_depth' : [5, 6, 7, 8, 9],
              'criterion' :['gini', 'entropy']
             }
tree_clas = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator=tree_clas, param_grid=param_grid, cv=5, verbose=True)
grid_search.fit(X_train_scaled, y_train)
#Fitting 5 folds for each of 90 candidates, totalling 450 fits
#GridSearchCV(cv=5, estimator=DecisionTreeClassifier(),
#              param_grid={'ccp_alpha': [0.1, 0.01, 0.001],
#                          'criterion': ['gini', 'entropy'],
#                          'max_depth': [5, 6, 7, 8, 9],
#                          'max_features': ['auto', 'sqrt', 'log2']},
#              verbose=True)
print(grid_search.best_params_)
accuracy = grid_search.best_score_ *100
print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy) )
# {'ccp_alpha': 0.1, 'criterion': 'gini', 'max_depth': 5, 'max_features': 'auto'}
# Accuracy for our training dataset with tuning is : 93.78%
model_train_test(DecisionTreeClassifier(ccp_alpha=0.1,criterion='gini',max_depth=5,max_features='auto'))
#HyperTension
X=data.drop('HighChol',axis=1)
y=data['HighChol']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=105,stratify=y)

print(X_train.shape)
print(X_test.shape)

# (49484, 10)
# (21208, 10)

#Scaling the data so that comparatively larger values do not make the model biased
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

accuracy_model=pd.DataFrame(columns=['Model','Accuracy'])

#Testing Different Models¶
#LOGISTIC REGRESSION
grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10)
model_train_test(logreg_cv)
#GAUSSIAN NAIVE BAYES
gnb=GaussianNB()
model_train_test(gnb)
#BERNOULLI NAIVE BAYES
bnb=BernoulliNB()
model_train_test(bnb)

#K-NEAREST NEIGHBOURS
knn = KNeighborsClassifier()

k_range = list(range(1, 20))
param_grid = dict(n_neighbors=k_range)

grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False, verbose=1)
grid_search = grid.fit(X_train_scaled, y_train)
#Fitting 10 folds for each of 19 candidates, totalling 190 fits
print(grid_search.best_params_)
accuracy = grid_search.best_score_ *100
print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy) )
#{'n_neighbors': 19}
#Accuracy for our training dataset with tuning is : 73.49%
model_train_test(KNeighborsClassifier(n_neighbors=19))
#DECISION TREE
param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
              'ccp_alpha': [0.1, .01, .001],
              'max_depth' : [5, 6, 7, 8, 9],
              'criterion' :['gini', 'entropy']
             }
tree_clas = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator=tree_clas, param_grid=param_grid, cv=5, verbose=True)
grid_search.fit(X_train_scaled, y_train)
#Fitting 5 folds for each of 90 candidates, totalling 450 fits
# GridSearchCV(cv=5, estimator=DecisionTreeClassifier(),
#              param_grid={'ccp_alpha': [0.1, 0.01, 0.001],
#                          'criterion': ['gini', 'entropy'],
#                          'max_depth': [5, 6, 7, 8, 9],
#                          'max_features': ['auto', 'sqrt', 'log2']},
#              verbose=True)

print(grid_search.best_params_)
accuracy = grid_search.best_score_ *100
print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy) )

model_train_test(DecisionTreeClassifier(ccp_alpha=0.001,criterion='entropy',max_depth=8,max_features='log2'))



#Diabetes
#Splitting the data into input data features and target¶
X=data.drop('Diabetes',axis=1)
y=data['Diabetes']

#Train-Test-Split and Scaling¶
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=105,stratify=y)
#Scaling the data so that comparatively larger values do not make the model biased
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
accuracy_model=pd.DataFrame(columns=['Model','Accuracy'])

#Testing Different Models
#LOGISTIC REGRESSION
grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10)
model_train_test(logreg_cv)
#GAUSSIAN NAIVE BAYES
gnb=GaussianNB()
model_train_test(gnb)
#BERNOULLI NAIVE BAYES
bnb=BernoulliNB()
model_train_test(bnb)

#K-NEAREST NEIGHBOURS
knn = KNeighborsClassifier()

k_range = list(range(1, 20))
param_grid = dict(n_neighbors=k_range)

grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False, verbose=1)
grid_search = grid.fit(X_train_scaled, y_train)
#Fitting 10 folds for each of 19 candidates, totalling 190 fits
print(grid_search.best_params_)
accuracy = grid_search.best_score_ *100
print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy) )
model_train_test(KNeighborsClassifier(n_neighbors=19))
#DECISION TREE
param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
              'ccp_alpha': [0.1, .01, .001],
              'max_depth' : [5, 6, 7, 8, 9],
              'criterion' :['gini', 'entropy']
             }
tree_clas = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator=tree_clas, param_grid=param_grid, cv=5, verbose=True)
grid_search.fit(X_train_scaled, y_train)

#
# Fitting 5 folds for each of 90 candidates, totalling 450 fits
# GridSearchCV(cv=5, estimator=DecisionTreeClassifier(),
#              param_grid={'ccp_alpha': [0.1, 0.01, 0.001],
#                          'criterion': ['gini', 'entropy'],
#                          'max_depth': [5, 6, 7, 8, 9],
#                          'max_features': ['auto', 'sqrt', 'log2']},
#              verbose=True)

print(grid_search.best_params_)
accuracy = grid_search.best_score_ *100
print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy) )

#
# {'ccp_alpha': 0.001, 'criterion': 'entropy', 'max_depth': 9, 'max_features': 'sqrt'}
# Accuracy for our training dataset with tuning is : 72.62%
model_train_test(DecisionTreeClassifier(ccp_alpha=0.001,criterion='gini',max_depth=5,max_features='auto'))





#
# percent_missing = round(df.isnull().sum() * 100 / len(df),2)
# missing_value_df = pd.DataFrame({'column_name': df.columns,
#                                  'percent_missing': percent_missing})
# #print(missing_value_df)
#
#
# #Defining a function to standardize the model testing process
#

