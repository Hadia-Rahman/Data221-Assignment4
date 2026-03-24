# Question 5
# Import statements
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# loading and training
data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# decision tree
tree = DecisionTreeClassifier(criterion="entropy",random_state=42)
tree.fit(X_train, y_train)
# training and test prediction for Decision tree
y_train_prediction = tree.predict(X_train)
y_test_prediction = tree.predict(X_test)
train_accuracy = accuracy_score(y_train, y_train_prediction)
test_accuracy = accuracy_score(y_test, y_test_prediction)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# neural net
Neaural_network_for_data = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42, activation ='relu')
Neaural_network_for_data.fit(X_train_scaled, y_train)

# training and test predictions for neural net
train_predictions = Neaural_network_for_data.predict(X_train_scaled)
test_predictions = Neaural_network_for_data.predict(X_test_scaled)

Decision_tree_confusion_matrix = confusion_matrix(y_test, y_test_prediction) # confusion matrix for decision tree
Neaural_network_confusion_matrix = confusion_matrix(y_test, test_predictions) # confusion matrix for neural net
print("Confusion Matrix for decision tree: ")
print(Decision_tree_confusion_matrix)
print("Confusion Matrix for Neural Network: ")
print(Neaural_network_confusion_matrix)

'''

QUESTION: Which model would you prefer for this task?
ANSWER: The model I prefer for this task would be the neural network because it is able to generalize and learn more complex patters better.
Additionally due to its ability to learn the patterns better it will be able to make more accurate predictions on unseen data.

QUESTION: What is one advantage and one limitation for each model?
ANSWER: 
    - Neural Network:
        - Advantage: able to learn more complex patters better.
        - Disadvantage: Difficult to understand or visualize how its decisions are made 
    - Decision Tree:
        - Advantage: Easy to interpret and visualize
        - Disadvantage: If the tree becomes too complex it can lead to overfitting 

'''