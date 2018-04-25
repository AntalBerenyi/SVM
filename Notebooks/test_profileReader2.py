from unittest import TestCase
from scripts.profile_reader2 import ProfileReader, TrainedSystemMarkers
import pandas as pd

class TestProfileReader2(TestCase):

    def test_get_profile_default(self):
        pr = ProfileReader(
            data_file='data\\Final_Berg JBS 2013 Supplemental Table 3_For SVM14Dec2017.xlsx',
            mechanism_file='data\\Final_Berg JBS 2013 Supplemental Table 3_For SVM14Dec2017 - Mechanisms.xlsx')
        df = pr.get_profile()
        self.assertEqual(df.shape, (327, 84))


    def test_get_profile_prof(self):
        pr = ProfileReader(
            data_file='data\\Final_Berg JBS 2013 Supplemental Table 3_For SVM14Dec2017.xlsx',
            mechanism_file='data\\Final_Berg JBS 2013 Supplemental Table 3_For SVM14Dec2017 - Mechanisms.xlsx')
        df = pr.get_profile(index='prof')
        self.assertEqual(df.shape, (327, 84))

    def test_get_profile_mech(self):
        pr = ProfileReader(
            data_file='data\\Final_Berg JBS 2013 Supplemental Table 3_For SVM14Dec2017.xlsx',
            mechanism_file='data\\Final_Berg JBS 2013 Supplemental Table 3_For SVM14Dec2017 - Mechanisms.xlsx')
        df = pr.get_profile(index=['mech'])
        self.assertEqual(df.shape, (327, 84))


    def test_get_mechanisms(self):
        pr = ProfileReader(
            data_file='data\\Final_Berg JBS 2013 Supplemental Table 3_For SVM14Dec2017.xlsx',
            mechanism_file='data\\Final_Berg JBS 2013 Supplemental Table 3_For SVM14Dec2017 - Mechanisms.xlsx')
        ls = pr.get_mechanisms()
        self.assertEqual(len(ls), 28)


    def test_get_agents(self):
        pr = ProfileReader(
            data_file='data\\Final_Berg JBS 2013 Supplemental Table 3_For SVM14Dec2017.xlsx',
            mechanism_file='data\\Final_Berg JBS 2013 Supplemental Table 3_For SVM14Dec2017 - Mechanisms.xlsx')
        ls = pr.get_agents()
        self.assertEqual(len(ls), 88)
        print(ls)

    def test_get_profile_names(self):
        pr = ProfileReader(
            data_file='Validation/Berg JBS 2013 Supplemental Table 5 - Profiles.xlsx')
        print(pr.get_profile_names())


    def test_check_target_sm(self):

        data_file = 'Validation/Missing Column.xlsx'
        df = pd.read_excel(data_file)
         # keep only System:marker.
        sm_columns = [s.replace('SMC_IL-1b/TNF-a/IFN-g_24:', 'CASMC_HCL_IL-1b/TNF-a/IFN-g_24:') for s in df.columns.values[1:]]
        tsm = set(TrainedSystemMarkers()())
        ssm = set(sm_columns)
        cols_not_in_target = list(tsm - ssm)
        print(cols_not_in_target)

