# Question 4
# Import statements
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# loading the data
data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # training with stratify
scaler = StandardScaler() #Creates a scaler

X_train_scaled = scaler.fit_transform(X_train) # fitting scaler on training data
X_test_scaled = scaler.transform(X_test) # applying the same scaler on testing data

Neaural_network_for_data = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42, activation ='relu') # creating a hidden layer
Neaural_network_for_data.fit(X_train_scaled, y_train)
# training the model and making predictions
train_predictions = Neaural_network_for_data.predict(X_train_scaled)
test_predictions = Neaural_network_for_data.predict(X_test_scaled)
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)
# display
print("Train Accuracy: ", train_accuracy)
print("Test Accuracy: ", test_accuracy)

'''
QUESTION: Why is feature scaling important for neural networks?
ANSWER: Feature scaling is important fo neural networks because it makes it so that all input features are at a similar scale which leads to faster
training and model accuracy. Feature scaling allows the model to consider every feature and not just the dominant one. This allows the neural net to 
recognize the patterns in the smaller features as well so everything is considered equally or close.  

QUESTION: What does an epoch represent during neural network training?
ANSWER: During neural network training an epoch represents a full pass through the training dataset. Every epoch allows the model to view the training dataset and
improve and update its weights each time allowing it to make more accurate predictions.  

'''