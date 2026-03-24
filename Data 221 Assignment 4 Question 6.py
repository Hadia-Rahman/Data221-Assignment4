# QUESTION 6
# Import statements
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
# Loading data
(X_train , y_train ) , ( X_test , y_test ) = fashion_mnist.load_data()

# convert to 0s and 1s
X_train = X_train/255.0
X_test = X_test/255.0

# reshaping with requirements for Conv2D
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
model = Sequential() # create model
# neural net
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(optimizer="adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
model.fit(X_train, y_train, epochs=15)
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy: ", test_acc)

'''

QUESTION: Why are CNNs generally preferred over fully connected networks for image data?
ANSWER: CNNs are generally preferred over full connected neural networks for image data because neural networks usually expect 
a vector input so you must flatten the image meaning you would destroy the spatial layout and it would require too many parameters.
Hence the CNN would be better as it leads to local connections, smaller pattern detectors(edges,corners,textures), weight sharing and
feature maps. 

QUESTION: What are convolution layers learning in this task? 
ANSWER: In this task convolution layers are learning the smaller pattern detections which allow it to distinguish the features of
different clothing items which allows the model to differentiate between classes like coat and shirt. The more layers the more it is 
able to distinguish the diffrent features. 

'''