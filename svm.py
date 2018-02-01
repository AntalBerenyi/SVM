import pandas as pd
from Notebooks.random_profiles import RandomProfileGenerator

def run():
    """
    :rtype: None
    """
    print('hello svm!')

    rpg = RandomProfileGenerator(envelope_file='Notebooks\\SigEnvelopeFile.xml', data_file='Notebooks\\SVMData.xlsx')

    df = rpg.get_random_profiles(prof_num=12,envelope=RandomProfileGenerator.PROFILE,conf=RandomProfileGenerator._95, dist='randn')

    print(df)


if __name__ == '__main__':
    run()
