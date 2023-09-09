import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import sklearn

"""
The next code implements a machine learning framework to classify if a person is obese or not
It uses tensor flow to implement a neural network of 2 layers that uses as output a sigmoid activation function.

Model specifications and hyperparameters:
    * As it is a binary classification the best output activation function is sigmoid, which returns a probbaility between 0 and 1.
    * For the hidden layer it uses 'relu' activaion function since it is a popular an effective function.
    * The amount of units in the hidden layers and the amount of total layers was decided through different tries. 
      The number of units that returned the best accuracy and less overfitting was the chosen.
    * For the output layer only one unit is needed since it is a binary classification.
    * The optimizer adam sicne it converges fas, is efficient and correct bias.
    * Since the classification is binary, the best loss to use is binary crossentropy.
    * The amount of epochs was decided through different tries. The simpler the model was, the better results it returned.

Generalization:
    * Once the model is running, all the epochs will be displayed with its corresponding accuracy and loss.
      As it is shown, during each epoch the accuracy indeed improves and the loss decreases.
      Therefore, we can see the model generalizes.

"""

data = pd.read_csv('Person_Gender_Height_Weight_Index.csv') #Change to your own path

data['obese'] = (data.Index >= 4).astype('int')
y = (data['obese'] == 1).astype(int)

# The Gender column is categorical, so we have to encode it. 
# This encoding adds 1 column of 0 or 1 if the person is Male, afterwards it drops the original column

data['is_Male'] = (data['Gender'] == 'Male').astype(int)
data.drop(['Index', 'Gender'], axis = 1, inplace = True)
X = data

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features using the training data. This is important for Neural Networks because it improves the model training speed and helps to find global minima.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


"""
MODEL:
"""
#Creates the Neural Network Model
model = keras.Sequential()
model.add(keras.layers.Dense(units=20, activation='relu', input_shape=(X_train.shape[1],)))
model.add(keras.layers.Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# The model automatically separates a validation set.
model.fit(X_train, y_train, epochs=10, validation_split=0.3, verbose=2) 
model.evaluate(X_test, y_test)


"""
PREDICTIONS:
"""
# The activation function 'sigmoid' outputs continuous values between 0 and 1, 
# therefore we have to convert the probabilities in 1 or 0 using 0.5 as threshold. 
predictions = (model.predict(X_test) > 0.5).astype(int)


"""
METRICS:
"""
accuracy = sklearn.metrics.accuracy_score(y_test, predictions)
precision = sklearn.metrics.precision_score(y_test, predictions)
recall = sklearn.metrics.recall_score(y_test, predictions)
f1_score = sklearn.metrics.f1_score(y_test, predictions)
roc_auc = sklearn.metrics.roc_auc_score(y_test, predictions)

results_df = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'])
results_df = results_df.append({    'Accuracy': accuracy,
                                    'Precision': precision,
                                    'Recall': recall,
                                    'F1-Score': f1_score,
                                    'ROC AUC': roc_auc}, ignore_index=True)

print(results_df)
confusion_matrix = sklearn.metrics.confusion_matrix(y_test, predictions)
cm_display = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()
