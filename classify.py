from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


# multi-class SVM classifier that implements the "one-against-one" approach
# decision_function_shape:
#   "ovr": one-vs-rest
#   "ovo": one-vs-one
# kernel: "rbf", "linear", "poly", "sigmoid", "precomputed"
# class_weight: dictionary of the form {class_label : value}, where value is a fp number > 0 that sets the parameter
#               c of class class_label to c * value.
#               Sets the parameter to "balanced" to sets the weights to n_samples / (n_classes * np.bincount(y))
def multiclassSVM(X, Y, decision_function_shape="ovr", kernel="rbf", class_weight=None):
    clf = svm.SVC(decision_function_shape=decision_function_shape, kernel=kernel, class_weight=class_weight)
    clf.fit(X, Y)
    return clf


# multi-class SVM classifier that implements the "one-vs-the-rest" approach
# class_weight: dictionary of the form {class_label : value}, where value is a fp number > 0 that sets the parameter
#               c of class class_label to c * value.
#               Sets the parameter to "balanced" to sets the weights to n_samples / (n_classes * np.bincount(y))
def multiclassLinearSVM(X, Y, class_weight=None):
    clf = svm.LinearSVC(class_weight=class_weight)
    clf.fit(X, Y)
    return clf


# random forest classifier
#   num_trees: number of trees in the forest
# class_weight: dictionary of the form {class_label : value}, where value is a fp number > 0 that sets the parameter
#               c of class class_label to c * value.
#               Sets the parameter to "balanced" to sets the weights to n_samples / (n_classes * np.bincount(y))
#               Sets the parameter to "balanced_subsample" to compute the weights on the base of the bootstrap sample
#               for every tree grown
def randomForest(X, Y, num_trees=100, class_weight=None):
    clf = RandomForestClassifier(n_estimators=num_trees, class_weight=class_weight)
    clf.fit(X, Y)
    return clf
