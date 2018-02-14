from unittest import TestCase
from Notebooks.scripts.profile_reader import ProfileReader
import pandas as pd

class TestProfileReader(TestCase):

    def test_parse_profiles_bad_file(self):
        pr = ProfileReader(data_file='Notebooks\\SVMDataxxxx.xlsx')
        df = pr.parse_profiles()
        # print(df)
        self.assertTrue('No such file or directory' in df)

    def test_parse_profiles(self):
        pr = ProfileReader(data_file='Notebooks\\data\\Final_Berg JBS 2013 Supplemental Table 3_For SVM14Dec2017.xlsx',
                           mechanism_file='Notebooks\\data\\Final_Berg JBS 2013 Supplemental Table 3_For SVM14Dec2017 - Mechanisms.xlsx')
        df = pr.parse_profiles()
        print(df.iloc[288:315, 0:3])

    def test_v_line_positions(self):
        pr = ProfileReader(data_file='Notebooks\\SVMData.xlsx')
        x, x_labels, v_line_positions = pr.v_line_positions()

        print(x)
        print(x_labels)
        print(v_line_positions)



    def test_parse_profiles_mTor(self):
        pr = ProfileReader(data_file='Notebooks\\SVMData.xlsx',  mechanism_file='Notebooks\\data\\Final_Berg JBS 2013 Supplemental Table 3_For SVM14Dec2017 - Mechanisms.xlsx')
        df = pr.parse_profiles()
        print(df.loc['mTOR inhibitor'].index.get_level_values(0))





