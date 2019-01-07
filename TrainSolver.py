import numpy as np
import sys
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import pickle


# def only_check(model_file,X):
#     model = load_model(model_file)
#     print model.predict_score(X[4000:4001,:])
#     print
#     print X[4000:4001,:]
#     exit()


def main(feature_vec = "vec_file.txt", model_file = "saved_model_short"):

    X_train, y_train = load_svmlight_file(feature_vec)
    print ("loaded")
    #only_check(model_file,X_train)
    clf = LinearSVC(penalty='l2',verbose=True,C=1)
    model = clf.fit(X_train,y_train)
    pickle.dump(clf, open(model_file, 'wb'))
    return model_file

if __name__ == '__main__':
    main()

