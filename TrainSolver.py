import numpy as np
import sys
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import pickle


# def only_check(model_file,X):
#     model = load_model(model_file)
#     print model.predict_score(X[4000:4001,:])
#     print
#     print X[4000:4001,:]
#     exit()
def plot_coefficients(classifier, feature_names = 0, top_features=20):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    # feature_names = np.array(feature_names)
    # plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha=’right’)
    plt.show()




def main(feature_vec="vec_file.txt", model_file="saved_model_short"):
    X_train, y_train = load_svmlight_file(feature_vec)
    print("loaded")
    # only_check(model_file,X_train)
    clf = LinearSVC(penalty='l2', verbose=False, C=.5)
    model = clf.fit(X_train, y_train)
    pickle.dump(clf, open(model_file, 'wb'))
    # plot_coefficients(model)
    # exit()
    return model_file


if __name__ == '__main__':
    main()
