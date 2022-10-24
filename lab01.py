import os
import numpy as np
from threading import Thread
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedKFold

FOLDER_PATH = "datasets-20221010/"

# Function to read form file lists
def file_lists():
    file_lists = []

    for file in os.listdir("datasets-20221010/"):
        file_lists.append(file)

    file_lists.sort()
    return file_lists


# FUnction to calculate model
def model(file_lists):
    dataset = np.genfromtxt((FOLDER_PATH + file_lists), delimiter=",")

    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=.30,
        random_state=1234
    )
    
    kf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1234)
    
    # Gaussian
    clf = GaussianNB()
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        scores.append(accuracy_score(y_test, predict))

    mean_score = np.mean(scores)
    std_score = np.std(scores)

    
    # Decision Tree
    dtc = DecisionTreeClassifier()
    scores = []
    
    for train_index, test_index, in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        dtc.fit(X_train, y_train)
        predict = dtc.predict(X_test)
        scores.append(accuracy_score(y_test, predict))

    mean_score_dtc = np.mean(scores)
    std_score_dtc = np.std(scores)

    print("Accuracy score: %.3f (%.3f), %.3f (%.3f)" % (mean_score, std_score, mean_score_dtc, std_score))
    return mean_score, std_score, mean_score_dtc, std_score_dtc


def calculating(file):
    threads = []
    for item in file:
        t = Thread(target=model, args = (item,))
        threads.append(t)
        t.start()
    return threads, t
    

if __name__ == '__main__':
    file = file_lists()
    print(file)

    threads, item = calculating(file)
    for t in threads:        
        t.join()
