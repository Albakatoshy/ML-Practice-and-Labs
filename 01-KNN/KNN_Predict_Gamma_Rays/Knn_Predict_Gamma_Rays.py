# %% [markdown]
# K-Nearest Neighbors Classification Predict either **gammas (signal)** or **hadrons (background)** using the MAGIC gamma telescope dataset.

# %%
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

#test our model
from sklearn.metrics import confusion_matrix , accuracy_score ,precision_score ,recall_score, f1_score

# %%
dataSet = pd.read_csv(r'KNN_Predict_Gamma_Rays\magic04.data', header=None)
dataSet

#12332 gamma events and 6688 hadron events.  



# %%
#split data into features and labels
X = dataSet.iloc[: , :-1]
Y = dataSet.iloc[: , -1]

Y.value_counts()


# %%
#Random Under Sampling
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(sampling_strategy=1)
X_resampled_RUS , Y_resampled_RUS = rus.fit_resample(X , Y)
X_resampled_RUS.value_counts()

# %%
#Random Over Sampling
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(sampling_strategy=1)
X_resampled_ROS , Y_resampled_ROS = ros.fit_resample(X , Y)
Y_resampled_ROS.value_counts()


# %%
#using Pandas to balance the dataset

gamma_data = dataSet[dataSet[10] == 'g']
hadron_data = dataSet[dataSet[10] == 'h']

gamma_data_underSampled = gamma_data.sample(n=len(hadron_data) , random_state=0)
balanced_data = pd.concat([gamma_data_underSampled , hadron_data])
balanced_data[10].value_counts()
balanced_data

# %%
#split data into training and testing sets
X_train , X_temp , Y_train , Y_temp = train_test_split(X_resampled_RUS , Y_resampled_RUS , test_size=0.3 , random_state=0)
X_test , X_val , Y_test , Y_val = train_test_split(X_temp , Y_temp , test_size=0.5 , random_state=0)

# %%
#Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
X_val = sc_X.transform(X_val)

# %%
K_values = [3 , 5 , 7 , 9 , 11]
for K in K_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=K , p = 2 , metric='euclidean')
    knn_classifier.fit(X_train , Y_train)
    
    #first predict on the validation set:
    Y_val_pred = knn_classifier.predict(X_val)

    #Report all of your trained model accuracy, precision, recall and f-score as well as confusion matrix
    print(f'K = {K} : ')
    print("accuracy_score : " , accuracy_score(Y_val , Y_val_pred))
    print("percision_score : " , precision_score(Y_val , Y_val_pred , pos_label='g'))
    print("recall_score : " , recall_score(Y_val , Y_val_pred , pos_label='g'))
    print("f1_score : " , f1_score(Y_val , Y_val_pred , pos_label='g'))
    print("confusion_matrix : \n" , confusion_matrix(Y_val , Y_val_pred))

    
    

# %%
# predict the test set results
classifier = KNeighborsClassifier(n_neighbors=7 , p = 2 , metric='euclidean')
Y_test_pred = classifier.fit(X_train , Y_train)
Y_test_pred = classifier.predict(X_test)
print("accuracy_score : " , accuracy_score(Y_test , Y_test_pred))
print("percision_score : " , precision_score(Y_test , Y_test_pred , pos_label='g'))
print("recall_score : " , recall_score(Y_test , Y_test_pred , pos_label='g'))
print("f1_score : " , f1_score(Y_test , Y_test_pred , pos_label='g'))
print("confusion_matrix : \n" , confusion_matrix(Y_test , Y_test_pred))



# %%



