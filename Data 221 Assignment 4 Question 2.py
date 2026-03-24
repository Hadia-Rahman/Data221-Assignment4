# Question 2
# import statements
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# loading the data set
data = load_breast_cancer()
X = data.data
y = data.target

# training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# creating a decision tree classifier
tree = DecisionTreeClassifier(criterion="entropy",random_state=42)
tree.fit(X_train, y_train) # fitting
y_train_prediction = tree.predict(X_train)
y_test_prediction = tree.predict(X_test)
train_accuracy = accuracy_score(y_train, y_train_prediction)
test_accuracy = accuracy_score(y_test, y_test_prediction)
# display
print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

'''

QUESTION: WHAT DOES ENTROPY REPRESENT IN THE CONTEXT OF DECISION TREES?
ANSWER: In the context of decision tress entropy represents the impurities disorders or uncertainty in a dataset. 
In a decision tree the goal is to achieve a pure node meaning all decisions belong to one class. The entorpy measures how well
a feature split achieve that.

QUESTION: DOES THE OBSERVED RESULT SUGGEST OVERFITTING OR GOOD GENERALIZATION?
ANSWER: The observed result suggests overfitting because the train accuracy 1.0 is higher then the test accuracy 0.91228...
meaning the model the training data but not as well on the unseen data.  

'''