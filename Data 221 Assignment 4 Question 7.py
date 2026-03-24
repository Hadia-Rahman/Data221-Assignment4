# Question 7
# Import statements
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# loading given in the assignment
(X_train , y_train ) , ( X_test , y_test ) = fashion_mnist.load_data() # loads the dataset

# We divide by 255 to normalize the pixels to 0s and 1s
X_train = X_train/255.0
X_test = X_test/255.0

# reshaping with requirements for Conv2D
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
model = Sequential() # Creates neural net

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1))) # feature detection
model.add(MaxPooling2D(pool_size=(2, 2))) # Makes the image smaller
model.add(Flatten()) # 2D to 1D

model.add(Dense(10, activation='softmax')) # adds output layer for 10 units/classes

model.compile(optimizer="adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"]) # compile

model.fit(X_train, y_train, epochs= 5) # 15 epochs

test_loss, test_acc = model.evaluate(X_test, y_test) # unseen data

print("Test Accuracy: ", test_acc) # accuracy

y_prediction_probabilities = model.predict(X_test)

y_prediction = np.argmax(y_prediction_probabilities, axis=1) # probabilities to class labels

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
# since the class names are numbers right now I had to go to the website to get the class names and put it in class_names

# confusion matrix image logic
confusion_matrix_for_data = confusion_matrix(y_test, y_prediction)
plt.imshow(confusion_matrix_for_data, cmap = "RdPu")
plt.title("Confusion Matrix for Fashion MNIST Dataset")
plt.colorbar()
plt.xticks(np.arange(10), class_names, rotation = 45)
plt.yticks(np.arange(10), class_names)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

# loop for confusion matrix image
for i in range (10):
    for j in range (10):
        plt.text(j, i, format(confusion_matrix_for_data[i, j]), ha="center", va="center", color="black")

plt.tight_layout()
plt.show()
misclassified_index = []
# loop for the misclassified images
for i in range(len(y_test)):
    if y_test[i] != y_prediction[i]:
        misclassified_index.append(i)
for i in range(3):
    atleast_3_misclassification = misclassified_index[i]
    plt.imshow(X_test[atleast_3_misclassification].reshape(28,28),cmap="gray")
    plt.title(f"True: {class_names[y_test[atleast_3_misclassification]]}, Predicted: {class_names[y_prediction[atleast_3_misclassification]]}")
    plt.axis("off")
    plt.show()

'''

QUESTION: What is one pattern you observed in the miscalculation?
ANSWER: One pattern I observed in the miscalculations was that images that looked very similar were being misclassified like 
shirt and coat this is due to the model being unable to distinguish similar features. 

QUESTION: What is one realistic method to improve the CNN performance
ANSWER: One realistic method to improve the CNN performance would be to add more convolutions layers because we know that the CNN 
is able to learn features like texture shape and edges adding more layers allows the model to learn more detailed features. Thus the model
will be able to distinguish similar features better. I also considered having more epochs because it represents a full pass through the 
training data which would allow the model to learn more features. However, I realized that adding more epochs may also result in overfitting and I already have 15
so I believe the convolution layers is a more realistic method.

'''