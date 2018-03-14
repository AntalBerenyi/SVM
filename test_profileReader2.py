from unittest import TestCase
from Notebooks.scripts.profile_reader2 import ProfileReader

class TestProfileReader2(TestCase):

    def test_get_profile_default(self):
        pr = ProfileReader(
            data_file='Notebooks\\data\\Final_Berg JBS 2013 Supplemental Table 3_For SVM14Dec2017.xlsx',
            mechanism_file='Notebooks\\data\\Final_Berg JBS 2013 Supplemental Table 3_For SVM14Dec2017 - Mechanisms.xlsx')
        df = pr.get_profile()
        self.assertEqual(df.shape, (327, 84))


    def test_get_profile_prof(self):
        pr = ProfileReader(
            data_file='Notebooks\\data\\Final_Berg JBS 2013 Supplemental Table 3_For SVM14Dec2017.xlsx',
            mechanism_file='Notebooks\\data\\Final_Berg JBS 2013 Supplemental Table 3_For SVM14Dec2017 - Mechanisms.xlsx')
        df = pr.get_profile(index='prof')
        self.assertEqual(df.shape, (327, 84))


    def test_get_mechanisms(self):
        pr = ProfileReader(
            data_file='Notebooks\\data\\Final_Berg JBS 2013 Supplemental Table 3_For SVM14Dec2017.xlsx',
            mechanism_file='Notebooks\\data\\Final_Berg JBS 2013 Supplemental Table 3_For SVM14Dec2017 - Mechanisms.xlsx')
        ls = pr.get_mechanisms()
        self.assertEqual(len(ls), 28)


    def test_get_agents(self):
        pr = ProfileReader(
            data_file='Notebooks\\data\\Final_Berg JBS 2013 Supplemental Table 3_For SVM14Dec2017.xlsx',
            mechanism_file='Notebooks\\data\\Final_Berg JBS 2013 Supplemental Table 3_For SVM14Dec2017 - Mechanisms.xlsx')
        ls = pr.get_agents()
        self.assertEqual(len(ls), 88)
        print(ls)