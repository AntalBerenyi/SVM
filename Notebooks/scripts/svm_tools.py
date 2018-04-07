import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score
from sklearn.model_selection import learning_curve, ShuffleSplit, StratifiedShuffleSplit, _validation
from sklearn.externals import joblib
from scripts.profile_reader2 import ProfileReader, TargetProcessor

class RS:
    '''
    Random seed class. Make seed value globally accessible.
    '''
    seed = 42

    @staticmethod
    def set_seed(seed):
        RS.seed = seed

    @staticmethod
    def get_seed():
        return RS.seed


def grid_search_svm(X, y, scorer=precision_score, cv='default', parameters= {'C': [0.001, 0.01, 0.1, 1.0, 10, 100, 1000]}):
    """
    GridSearch and return the best estimator
    :param X: The feature set
    :param y: The labels
    :param scorer: accuracy_score|precision_score|recall_score. The sklearn.metrics scorer to build the GridSearch object.
    :param cv : str, int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, ShuffleSplit(X.shape[0], n_iter=10, test_size=0.2)
          - 'default', to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    :return: The SVM classifier tuned to to the best C-value.
    """
    # Initialize the classifier
    clf = SVC(kernel='linear')

    scorer = make_scorer(scorer)

    if cv is None:
        cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=RS.get_seed())

    if cv == 'default':
        cv = None

    # Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
    grid_obj = GridSearchCV(clf, param_grid=parameters, scoring=scorer, cv=cv)

    # Fit the grid search object to the training data and find the optimal parameters using fit()
    grid_fit = grid_obj.fit(X, y)

    # Get the estimator
    return grid_fit.best_estimator_


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), figure=True):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : str, int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, ShuffleSplit(X.shape[0], n_iter=10, test_size=0.2)
          - 'default', to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    if figure:
        plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    if cv is None:
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=RS.get_seed())

    if cv == 'default':
        cv = None

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, random_state=RS.get_seed())
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def train_svm(data_file='data/Final_Berg JBS 2013 Supplemental Table 3_For SVM14Dec2017.xlsx',
                       mechanism_file='data/Final_Berg JBS 2013 Supplemental Table 3_For SVM14Dec2017 - Mechanisms.xlsx',
              impute='group_mean', parameters={'C': [500]}, normalize='l2', prof_num=30, dump='svm_classifiers.pkl'):
    '''

    :param data_file:
    :param mechanism_file:
    :param impute:
    :param normalize:
    :param prof_num:
    :param dump:
    :return:
    '''
    pr = ProfileReader(data_file='data/Final_Berg JBS 2013 Supplemental Table 3_For SVM14Dec2017.xlsx',
                       mechanism_file='data/Final_Berg JBS 2013 Supplemental Table 3_For SVM14Dec2017 - Mechanisms.xlsx')

    mechs = pr.get_mechanism_count()['Mechanism']
    clfs = {}

    # ensure reprodicible results
    np.random.seed(442)
    for mech in mechs[0:]:
        # get training data with 100 negative class
        X, y = pr.get_x_y(mech=mech, impute='group_mean', normalize='l2', prof_num=30)
        X0 = X[y == 1]

        # Synthetic Minority Oversampling Technique. Bring the positive class numbers up to the random negative class.
        k_n = min(Counter(y)[1] - 1, 5)
        X, y = SMOTE(k_neighbors=k_n, kind='regular').fit_sample(X, y)

        best_clf = grid_search_svm(X, y, scorer=precision_score, parameters=parameters)

        clfs.update({mech: best_clf})

    if dump:
        joblib.dump(clfs, 'svm_classifiers.pkl')
    return clfs


def predict_svm(data_file, classifier_file='svm_classifiers.pkl', include_negative=False, pivot=True):
    pr = TargetProcessor(data_file=data_file)
    clfs = joblib.load(classifier_file)
    target = pr.get_target()

    ptable0 = pd.DataFrame({'Profile': [],
                            'Mechanism': [],
                            'Decision Value': [],
                            'Prediction': []})
    for mech, clf in clfs.items():
        pred = clf.predict(target)
        dv = clf.decision_function(target)
        ptable = pd.DataFrame({'Profile': target.index.values,
                               'Mechanism': [mech] * len(target),
                               'Decision Value': dv,
                               'Prediction': pred})
        ptable0 = pd.concat([ptable0, ptable])
        if ~include_negative:
            ptable0 = ptable0[ptable0.Prediction == 1]

    ptable0 = ptable0[['Profile', 'Mechanism', 'Decision Value', 'Prediction']].sort_values(
        by=['Profile', 'Decision Value'], ascending=[True, False])

    if pivot:
        ptable0 = ptable0.pivot(index='Profile', columns='Mechanism', values='Decision Value')


    return ptable0
