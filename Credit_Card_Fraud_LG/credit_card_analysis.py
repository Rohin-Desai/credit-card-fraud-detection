import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import os 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression




#Data importing
file_path = r"C:\Users\rohin\Desktop\ML learning from zero\Credit_Card_Fraud_LG\creditcard.csv"

df = pd.read_csv(file_path)

#Class that is my custom logistic regression model
class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate  # Step size for updating weights
        self.epochs = epochs  # Number of iterations
        self.weights = None  # Placeholder for weights
        self.bias = None  # Placeholder for bias

    def sigmoid(self, z):
    #"Sigmoid activation function to convert values into probabilties"
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, y, y_pred):
        "compute log loss"
        m = len(y)
        loss = (-1/m) * np.sum(y * np.log(y_pred) + (1-y)* np.log(1 - y_pred))
        return loss
    
    def fit(self, X, y):
        """ Train Logistic Regression using Gradient Descent """
        m, n = X.shape  # m = number of examples, n = number of features
        self.weights = np.zeros((n, 1))  # Initialize weights to zeros
        self.bias = 0  # Initialize bias to zero

        y = y.reshape(-1, 1)

        for epoch in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            dw = (1 / m) * np.dot(X.T, (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Compute loss for monitoring
            if epoch % 100 == 0:
                loss = self.compute_loss(y, y_pred)
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

    def predict(self, X):
        """ predict class labels (0 or 1)"""
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred_prob = self.sigmoid(linear_model)
        return np.array([1 if prob >= 0.5 else 0 for prob in y_pred_prob]).reshape(-1, 1)

#Ensures all features have similar scales
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])


#Time doesn't seem that useful
df.drop(columns=["Time"], inplace=True)

#Create X (data with only training variables)
X = df.drop(columns=["Class"])
#y is the output labels for X
y = df["Class"]


#Training test split 80/20, stratify makes it so there is an equal portion of fraud between the test and train data
X_train, x_test, Y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state= 42, stratify=y)

#this is my model set to a learning rate of 1 using my custom logistic regression
model = LogisticRegressionScratch(learning_rate=0.1, epochs=2000)

#calls fit function on training data and creates the logistic regression curve to fit
model.fit(X_train.to_numpy(), Y_train.to_numpy())

#calls predict function on the test set 
y_pred_test = model.predict(x_test.to_numpy())

#Scikit libary metrics on model
accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)

#output metrics
print(f"Custom Logistic Regression Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

#uses sk model to train and test the data
sklearn_model = LogisticRegression()
sklearn_model.fit(X_train, Y_train)

y_pred_sklearn = sklearn_model.predict(x_test)

#sks performance

accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
precision_sklearn = precision_score(y_test, y_pred_sklearn)
recall_sklearn = recall_score(y_test, y_pred_sklearn)
f1_sklearn = f1_score(y_test, y_pred_sklearn)

print(f"\nScikit-Learn Logistic Regression Results:")
print(f"Accuracy: {accuracy_sklearn:.4f}")
print(f"Precision: {precision_sklearn:.4f}")
print(f"Recall: {recall_sklearn:.4f}")
print(f"F1 Score: {f1_sklearn:.4f}")





