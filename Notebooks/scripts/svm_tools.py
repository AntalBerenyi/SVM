import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score
from sklearn.model_selection import learning_curve, ShuffleSplit, StratifiedShuffleSplit, _validation
from sklearn.externals import joblib
from scripts.profile_reader2 import ProfileReader, TargetProcessor
import seaborn as sns;

#sns.set()  # set to default
pd.options.display.max_rows = 500
pd.options.display.max_columns = 200



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


def grid_search_svm(X, y, scorer=precision_score, cv='default',
                    parameters={'C': [0.001, 0.01, 0.1, 1.0, 10, 100, 1000]}):
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


class SVM:
    def __init__(self, target_data_file='../targets/target.xlsx', classifier_file='svm_classifiers.pkl'):
        '''
        Initilize the
        :param target_data_file:
        :param classifier_file:
        '''
        self.predictions = None
        self.ptable = None
        self.target_data_file = target_data_file
        self.classifier_file = classifier_file
        self.clfs = None

    def train(self, training_data_file='data/Final_Berg JBS 2013 Supplemental Table 3_For SVM14Dec2017.xlsx',
              mechanism_file='data/Final_Berg JBS 2013 Supplemental Table 3_For SVM14Dec2017 - Mechanisms.xlsx',
              impute='group_mean', parameters={'C': [500]}, normalize='l2', dump='svm_classifiers.pkl',
              smote=True, prof_num=None):
        '''
        Train the SVM and generate the model.
        :param training_data_file: The training data file containg the mechanism profiles.
        :param mechanism_file: The file that containg information about the training data
        :param impute: str 'group_mean'. The method to impute missing training data points. See
        :param parameters: map default {'C': [500]}. The grids search training parameters.
        :param normalize: str 'l1', 'l2', 'max'. Normalize input profile vectors to unit length. See sklearn.preprocessing.Normalizer
        :param dump: str default 'svm_classifiers.pkl'. This is the file name for the trained model to be writen.
        :param smote: boolean True|False default True. If set to true, and prof_num is set to a number larger than the
                        training data positve class, oversample the positive class using SMOTE method.
        :param prof_num: int default None. The number of negative class profiles to generate. If None, the number of negative
                classes to be generated is the same as the positive class.
        :return:
        '''
        pr = ProfileReader(data_file=training_data_file,
                           mechanism_file=mechanism_file)
        mechs = pr.get_mechanism_count()['Mechanism']
        clfs = {}
        # ensure reprodicible results
        np.random.seed(442)
        for mech in mechs[0:]:
            # get training data with 100 negative class
            X, y = pr.get_x_y(mech=mech, impute=impute, normalize_method=normalize, prof_num=prof_num)
            # Synthetic Minority Oversampling Technique. Bring the positive class numbers up to the random negative class.
            if smote:
                k_n = min(Counter(y)[1] - 1, 5)
                X, y = SMOTE(k_neighbors=k_n, kind='regular').fit_sample(X, y)
            best_clf = grid_search_svm(X, y, scorer=precision_score, parameters=parameters)
            clfs.update({mech: best_clf})
        self.clfs = clfs
        if dump:
            joblib.dump(clfs, 'svm_classifiers.pkl')
        return clfs

    def predict(self, normalize='l2', remove_neg=True, write_to_file=True, heatmap=True):
        '''
        Use the results of train_svm to predict mechanism of a drug class. The
        :param normalize: The normalization method used to normalize the training data positive class.
        :return: DataFrame, DataFrame. The first table one has one profile per row and the predicted mechanisms in
                    columns. The second DataFrame has the predicted mechanisms all in a single column unpivoted.
        '''

        self.clfs = joblib.load(self.classifier_file)
        pr = TargetProcessor(data_file=self.target_data_file)
        target = pr.get_target()
        target = ProfileReader.normalize(target, normalize)

        ptable0 = pd.DataFrame({'Profile': [],
                                'Mechanism': [],
                                'Decision Value': [],
                                'Prediction': []})
        for mech, clf in self.clfs.items():
            pred = clf.predict(target)
            dv = clf.decision_function(target)
            ptable = pd.DataFrame({'Profile': target.index.values,
                                   'Mechanism': [mech] * len(target),
                                   'Decision Value': dv,
                                   'Prediction': pred})
            if remove_neg:
                ptable = ptable[ptable.Prediction == 1]
            ptable0 = pd.concat([ptable0, ptable])

        ptable0 = ptable0[['Profile', 'Mechanism', 'Decision Value', 'Prediction']].sort_values(
            by=['Profile', 'Decision Value'], ascending=[True, False])

        self.predictions = ptable0.pivot(index='Profile', columns='Mechanism', values='Decision Value')
        self.ptable = ptable0

        if write_to_file:
            fn = '../outputs/SVM prediction.xlsx'
            try:
                os.remove(fn)
            except OSError:
                pass
            self.predictions.to_excel(fn, index=True)

        return self.predictions

    def heatmap(self):
        '''
        Generate a heatmap of the predictions. Note that you should call predict before calling heatmap.

        Usage:
        svm = SVM()
        svm.predict()
        svm.heatmap()

        :return:
        '''
        pr = TargetProcessor(data_file=self.target_data_file, verbose=False)
        plt.figure(figsize=(20, pr.get_profile_count()));
        ax = sns.heatmap(self.predictions, annot=True, fmt=".2", cmap='YlGnBu', cbar=None);
        plt.show()
        return ax

    def get_predictions(self):
        return self.predictions

    def get_best_class(self):
        '''
        Get the list of best predictions for each profile.
        :return: DataFrame with profile, Mechanism and DV for the best prediction.
        '''
        p2 = []
        for prof, mec in self.predictions.idxmax(1).items():
            dv = self.ptable[(self.ptable.Profile == prof) & (self.ptable.Mechanism == mec)]['Decision Value'].values[0]
            a = [prof, mec, dv]
            p2.append(a)

        p2 = pd.DataFrame(p2)
        p2.sort_values(by=[1, 0])
        p2.reset_index(inplace=True, drop=True)
        p2.columns = ['Profile', 'Mechanism', 'DV']
        return p2