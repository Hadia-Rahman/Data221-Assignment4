# Question 3
# import statement
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# loading and training
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# decision tree
tree = DecisionTreeClassifier(criterion="entropy",random_state=42, max_depth=5)
tree.fit(X_train, y_train)
y_train_prediction = tree.predict(X_train)
y_test_prediction = tree.predict(X_test)
train_accuracy = accuracy_score(y_train, y_train_prediction)
test_accuracy = accuracy_score(y_test, y_test_prediction)
# display
print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

top_5_importances = tree.feature_importances_
top_features = []
top_values = []
indices_used = []
# using a for loop to calculate the top features
for k in range(5):
    max_value = -1 # never this number so we are able to use it
    min_value = -1
    for i in range(len(top_5_importances)): # looping through the tree importances
        if i in indices_used:
            continue
        elif top_5_importances[i] > max_value:
            max_value = top_5_importances[i] # s
            max_index = i
    top_features.append(feature_names[max_index])
    top_values.append(max_value)
    indices_used.append(max_index) # so we dont repeat

# display
print("Top 5 most important features:")
for i in range(5):
    print(top_features[i], top_values[i])

'''
QUESTION: How does controlling model complexity affect overfitting?
ANSWER: Controlling model complexity lowers overfitting. By adding max depth = 5 we were able to limit the model complexity 
down to 5 levels allowing the dataset to focus more on learning the general patterns rather than just memorizing the training data which
improves its overall performance on unseen data.  

QUESTION: How does feature importance contribute to the interpretability of decision trees?
ANSWER: Feature importance contributes to the interpretability of decision trees by allowing us to find which features have the greatest influence
on the models predictions. This allows us to figure out how the decision tree makes the decisions and what variables are most relevant.  

'''