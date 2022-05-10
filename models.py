#Visualisation Libraries
import matplotlib.pyplot as plt
import seaborn as sns
#Array and DataStructures libraries
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 101)

# Dataset is already loaded below
data = pd.read_csv("train.csv")
#Explore columns
data.columns = data.columns.str.replace("'","")
data.columns = data.columns.str.replace(" ","_")
data.columns;

"""### Dimensionality Reduction

I am manually reducing the number of columns to be used for training the model so as to simplify the dataset and produce a better fit predictive model
"""

#Reducing number of columns in dataset
train_data_reduced_patient = data.drop(['Eye','LogMAR_UDVA',
       'LogMAR_CDVA', 'Sphere', 'Cylinder', 'Axis', 'Keratometry_Unit_',
       'Flat_Keratometry', 'Steep_Keratometry_', 'Steep_Axis_',
       'Mean_Topograpjy_K', 'Topography_Cylinder', 'Central_Pachy',
       'Thinnest_pachy', 'Location_X_Axis', 'Location_Y_Axis',
       'SLE_Corneal_Scarring', 'SLE_Vogts_Striae_', 'Fleischers_ring_'], axis = 1)

#Reducing number of columns in dataset
train_data_reduced_pentacam= data.drop(['Gender', 'Race', 'General_Health_(requiring_Medication)', 'Atopy_',
       'Hypertension', 'Hayfever_', 'Known_Eye_History', 'Eye_Rubbing',
       'Family_History_KC', 'Primary_Optical_Aid', 'Eye', 'LogMAR_UDVA',
       'LogMAR_CDVA', 'SLE_Corneal_Scarring', 'SLE_Vogts_Striae_', 'Fleischers_ring_'], axis = 1)

train_data_reduced_all = data.drop([ 'Eye', 'LogMAR_UDVA',
       'LogMAR_CDVA', 'SLE_Corneal_Scarring', 'SLE_Vogts_Striae_', 'Fleischers_ring_'], axis = 1)

"""### Data Wrangling and Cleanup for Test Dataset

"""

# Dataset is already loaded below
test_data = pd.read_csv("test.csv")
#Explore columns
test_data.columns = test_data.columns.str.replace("'","")
test_data.columns = test_data.columns.str.replace(" ","_")
test_data = test_data.astype({"Race":'str', "Primary_Optical_Aid":'str'})
test_data.dtypes

"""### Dimensionality Reduction

"""

#Reducing number of columns in dataset
test_data_reduced_patient = test_data.drop(['Eye','LogMAR_UDVA',
       'LogMAR_CDVA', 'Sphere', 'Cylinder', 'Axis', 'Keratometry_Unit_',
       'Flat_Keratometry', 'Steep_Keratometry_', 'Steep_Axis_',
       'Mean_Topograpjy_K', 'Topography_Cylinder', 'Central_Pachy',
       'Thinnest_pachy', 'Location_X_Axis', 'Location_Y_Axis',
       'SLE_Corneal_Scarring', 'SLE_Vogts_Striae_', 'Fleischers_ring_'], axis = 1)

#Reducing number of columns in dataset
test_data_reduced_pentacam= test_data.drop(['Gender', 'Race', 'General_Health_(requiring_Medication)', 'Atopy_',
       'Hypertension', 'Hayfever_', 'Known_Eye_History', 'Eye_Rubbing',
       'Family_History_KC', 'Primary_Optical_Aid', 'Eye', 'LogMAR_UDVA',
       'LogMAR_CDVA', 'SLE_Corneal_Scarring', 'SLE_Vogts_Striae_', 'Fleischers_ring_'], axis = 1)

test_data_reduced_all = data.drop([ 'Eye', 'LogMAR_UDVA',
       'LogMAR_CDVA', 'SLE_Corneal_Scarring', 'SLE_Vogts_Striae_', 'Fleischers_ring_'], axis = 1)

"""#### Illustrating the correlation between all the attributes"""

correlations_patient = train_data_reduced_patient.corr()
fig, ax = plt.subplots(figsize=(10,10))
#making heatmap with customised visualisation settings
sns.heatmap(correlations_patient, vmax=1.0, center=0, fmt='.2f', square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
plt.show()

correlations_pentacam = train_data_reduced_pentacam.corr()
fig, ax = plt.subplots(figsize=(10,10))
#making heatmap with customised visualisation settings
sns.heatmap(correlations_pentacam, vmax=1.0, center=0, fmt='.2f', square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
plt.show()

# creating the barplot with customised visualisation settings
correlation_viz_patient = correlations_patient
correlation_viz_patient['feature'] = correlation_viz_patient.index

ix = (abs(train_data_reduced_patient.corr()).sort_values('label', ascending = False).index)
order = train_data_reduced_patient.loc[:, ix]

plt.figure(figsize=(6,6))
sns.barplot(x='feature', y="label", data=correlation_viz_patient, order=order)
plt.xlabel("Features")
plt.ylabel("Correlation")
plt.title("Features")
plt.xticks(rotation=90)
plt.show()

# creating the barplot with customised visualisation settings
correlation_viz_pentacam = correlations_pentacam
correlation_viz_pentacam['feature'] = correlation_viz_pentacam.index

ix = (abs(train_data_reduced_pentacam.corr()).sort_values('label', ascending = False).index)
order = train_data_reduced_pentacam.loc[:, ix]

plt.figure(figsize=(6,6))
sns.barplot(x='feature', y="label", data=correlation_viz_pentacam, order=order)
plt.xlabel("Features")
plt.ylabel("Correlation")
plt.title("Features")
plt.xticks(rotation=90)
plt.show()

x1train = train_data_reduced_patient.drop(['label'], axis = 1)
y1train = train_data_reduced_patient['label']
x1test = test_data_reduced_patient.drop(['label'], axis = 1)
y1test = test_data_reduced_patient['label']

x2train = train_data_reduced_pentacam.drop(['label'], axis = 1)
y2train = train_data_reduced_pentacam['label']
x2test = test_data_reduced_pentacam.drop(['label'], axis = 1)
y2test = test_data_reduced_pentacam['label']

x3train = train_data_reduced_all.drop(['label'], axis = 1)
y3train = train_data_reduced_all['label']
x3test = test_data_reduced_all.drop(['label'], axis = 1)
y3test = test_data_reduced_all['label']

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from yellowbrick.classifier import ROCAUC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import warnings

def CreateModel(model, xtrain, xtest, ytrain, ytest, Name):
    train_acc = model.score(xtrain, ytrain)
    test_acc= model.score(xtest, ytest)

    print('\n')
    print(f"{Name}")
    print(f"Train Accuracy = {train_acc*100} %")
    print(f"Test Accuracy = {test_acc*100} %")

    fig, ax = plt.subplots(figsize = (6,6))
    visualizer = ROCAUC(model)
    visualizer.fit(xtrain, ytrain)        # Fit the training data to the visualizer
    visualizer.score(xtest, ytest)        # Evaluate the model on the test data
    visualizer.finalize()   
    ax.set_xlim([-0.01, 1])
    ax.set_ylim([0.0, 1.05])

    disp = ConfusionMatrixDisplay.from_estimator(model.fit(xtrain, ytrain),xtest,ytest,cmap=plt.cm.Blues,normalize='true',)
    disp.ax_.set_title("Normalized Confusion matrix", fontsize=9)
    disp.ax_.set_xlabel("Predicted Class", fontsize=9)
    disp.ax_.set_ylabel("True Class", fontsize=9)
    plt.grid(False)
    plt.show()

    return 

MODEL1 = LogisticRegression(solver='lbfgs', max_iter=10000)
MODEL2 = KNeighborsClassifier(4)
MODEL3 = SVC(kernel="linear", C=0.025)
MODEL4 = DecisionTreeClassifier(max_depth=5)
MODEL5 = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
MODEL6 = MLPClassifier(alpha=1, max_iter=1000)
MODEL7 = AdaBoostClassifier()
MODEL8 = QuadraticDiscriminantAnalysis()                            

MODELS = [MODEL1, MODEL2, MODEL3, MODEL4, MODEL5, MODEL6, MODEL7, MODEL8]
MODELNAMES = ["LogisticRegression", "KNeighborsClassifier", "SVC", "DecisionTreeClassifier", "RandomForestClassifier", "MLPClassifier", "ADABoostClassifier", "QuadraticDiscriminantAnalysis"]

warnings.filterwarnings('ignore', category=DeprecationWarning)
for i in range(len(MODELS)):
    CreateModel(MODELS[i].fit(x1train, y1train), x1train, x1test, y1train, y1test, MODELNAMES[i])

for i in range(len(MODELS)):
    CreateModel(MODELS[i].fit(x2train, y2train), x2train, x2test, y2train, y2test, MODELNAMES[i])

for i in range(len(MODELS)):
    CreateModel(MODELS[i].fit(x3train, y3train), x3train, x3test, y3train, y3test, MODELNAMES[i])