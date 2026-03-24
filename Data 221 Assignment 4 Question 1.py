# Question 1
# Starter code given in assignment to load the dataset
from sklearn.datasets import load_breast_cancer
loading_breast_cancer_data_set = load_breast_cancer() # loading the dataset

X = loading_breast_cancer_data_set.data
y = loading_breast_cancer_data_set.target

print(X.shape)
print(y.shape)

count_malignant = 0
count_benign = 0
# using a for loop for count
for label_in_breast_cancer_dataset in y :
    if label_in_breast_cancer_dataset == 0 :
        count_malignant+= 1
    else:
        count_benign += 1
print({"Class malignant": count_malignant, "Class benign": count_benign}) # display

'''
QUESTION 1: IS THE DATASET BALANCED OR IMBALANCED?
ANSWER: After counting for malignant and benign I found that there was a total of 212 for the class malignant and 357
for the class benign thus this dataset is imbalanced.  

QUESTION 2: WHY IS CLASS BALANCE AN IMPORTANT CONSIDERATION FOR CLASSIFICATION MODELS
Class imbalance is an important consideration for classification models because it causes the model to predict well 
with the majority class but have bad performance with the minority. If the model only works well with the majority class it can 
also be considered a fairness issue because the accuracy may become biased or unequal. 
'''