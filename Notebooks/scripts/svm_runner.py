import numpy as np
import pandas as pd
from scripts.profile_reader2 import ProfileReader

class SVM():

    mech_ = []

    def predict(self, target_file="targets/ToxCast.xlsx"):
        pr = ProfileReader(data_file=target_file)

        result = pd.DataFrame({"Profile": "None", "DV": {"m1": 0.8, "m2": 0.45, "m3": 0.12}})
        return result

