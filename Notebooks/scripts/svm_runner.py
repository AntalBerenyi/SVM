import numpy as np
from scripts.profile_reader2 import ProfileReader

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set() # set to default
from sklearn.externals import joblib
from IPython.display import display
from scripts.profile_reader2 import ProfileReader, TargetProcessor


class SVM():

    mech_ = []

    def predict(self, target_file="targets/target.xlsx"):
        pr = TargetProcessor(data_file=data_file)

        clfs = joblib.load('svm_classifiers.pkl')
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
            # ptable0 = ptable0[ptable0.Prediction == 1]

        ptable0 = ptable0[['Profile', 'Mechanism', 'Decision Value', 'Prediction']].sort_values(
            by=['Profile', 'Decision Value'], ascending=[True, False])


        return result

