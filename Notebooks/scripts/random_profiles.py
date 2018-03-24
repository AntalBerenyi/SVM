import pandas as pd
import numpy as np
# https://github.com/martinblech/xmltodict
# conda install -c conda-forge xmltodict
import xmltodict as x2d
from numpy.random import rand
from numpy.random import randn



class RandomProfileGenerator:
    # indexes in xml to select profile type/confidence
    SCREENING = 0
    TRUSTED = 1
    PROFILE = 2
    _99 = 0
    _95 = 1

    def __init__(self, envelope_file='SigEnvelopeFile.xml', data_file='SVMData.xlsx', skip_cols=1):
        '''
        Constructor initializing generator.
        :param envelope_file: path to the significance envelope file
        :param data_file: Path to profile data file. Used for column names.
        :param skip_cols: skip the first skip_col columns
        '''
        with open(envelope_file) as f:
            xml_data = f.read()
            # parse xml string as a dictionary
            self.envelope_dict = x2d.parse(xml_data)

        if isinstance(data_file, str):
            # read SVM published profiles, and get system:marker values from there
            data = pd.read_excel(data_file)

        if isinstance(data_file, pd.DataFrame):
            data = data_file

        # sequence of system-markers in profile is the column names:
        self.system_markers = data.columns.values[skip_cols:]


    def parse_df(self):

        pass

    def get_random_profiles(self, prof_num=10, envelope=SCREENING, conf=_95, dist='rand'):
        """
        Generate random profiles within the specified envelope values
        :param prof_num:
        :param envelope: int, envelope type
        :param conf: float, confidence interval
        :return: DataFrame, each row a random profile
        """
        value_list = self.envelope_dict['java']['object']['void'][envelope]['object'][1]['void'][conf]['object']['void']
        # populate values from value_list into a dict, "system:marker':value
        envelope = self.get_envelope(envelope, conf)
        for item in value_list:
            envelope[item['string']] = item['float']

        #####################################
        # now generate random weak profiles #
        #####################################

        # ## Generate random Profiles
        # _rand_f = rand
        # _multip = 1
        # if dist == 'randn':
        #     # if randn specified use gaussian distribution
        #     _rand_f = randn
        #     # calculate sigma of normal dist
        #     # for .95 -> Z=1.96  .99 -> Z=2.576
        #     # (x - mu)/sigma = Z  ; x = 1; mu = 0
        #     if conf == self._95:
        #         _multip = 1 / 1.96
        #     if conf == self._99:
        #         _multip = 1 / 2.576

        random_profiles = []
        for rp in range(prof_num):
            random_profile = [self.random_scaled_value(float(envelope[sm]), dist, conf) for sm in self.system_markers]
            random_profiles.append(random_profile)

        rp_df = pd.DataFrame(random_profiles, columns=self.system_markers)

        return rp_df

    def random_scaled_value(self, env_value, dist, conf):
        """

        :param env_value:
        :param dist: if 'randn' specified use gaussian distribution. To get a uniform distribution use 'rand'.
        :param conf: Use this value to determine the width of the Gaussian distribution. _95 or _99
        :return:
        """

        val = 0
        if dist == 'randn':
            # if randn specified use gaussian distribution
            # calculate sigma of normal dist
            # for .95 -> Z=1.96  .99 -> Z=2.576
            # (x - mu)/sigma = Z  ; x = 1; mu = 0
            if conf == self._95:
                sigma = 1 / 1.96
            if conf == self._99:
                sigma = 1 / 2.576
            val = sigma * randn() * env_value
        if dist == 'rand':
            val = (rand() * 2 - 1) * env_value
        return val


    def get_envelope(self, envelope=SCREENING, conf=_95):

        value_list = self.envelope_dict['java']['object']['void'][envelope]['object'][1]['void'][conf]['object']['void']
        # populate values from value_list into a dict, "system:marker':value
        envelope = {}
        for item in value_list:
            envelope[item['string']] = item['float']

        return envelope

    def get_neg_class(self, envelope=SCREENING, conf=_95, dist='rand', prof_num=10):
        """
        Generate negative class for profiles. The system-readouts are in the same order as the data file specified.
        :param envelope: The envelope type
        :param conf: The confidence interval
        :param dist: random profile distribution type: 'rand' or 'randn'
        :param prof_num: The number of random profiled to generate.
        :return:
        """
        neg_values = self.get_random_profiles(prof_num=prof_num, envelope=envelope, conf=conf, dist=dist)
        neg_mech = pd.DataFrame({'mech': ['neg_class' for _ in range(prof_num)]})
        neg_class = pd.concat([neg_mech, neg_values], axis=1)
        neg_class.set_index('mech', inplace=True)
        return neg_class


