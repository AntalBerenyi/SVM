import pandas as pd
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

    def __init__(self, envelope_file='SigEnvelopeFile.xml', data_file='SVMData.xlsx'):
        '''
        Constructor initializing generator.
        :param envelope_file: path to the significance envelope file
        :param data_file: Path to profile data file. Used for column names.
        '''
        with open(envelope_file) as f:
            xml_data = f.read()
            # parse xml string as a dictionary
            self.envelope_dict = x2d.parse(xml_data)

        # read SVM published profiles, and get system:marker values from there
        data = pd.read_excel(data_file)

        # sequence of system-markers in profile is the column names:
        self.system_markers = data.columns.values[1:]


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

        ## Generate random Profiles
        _rand_f = rand
        _multip = 1
        if dist == 'randn':
            # if randn speccified use gaussian distribution
            _rand_f = randn
            # calculate sigma of normal dist
            # for .95 -> Z=1.96  .99 -> Z=2.576
            # (x - mu)/sigma = Z  ; x = 1; mu = 0
            if conf == self._95:
                _multip = 1 / 1.96
            if conf == self._99:
                _multip = 1 / 2.576

        random_profiles = []
        for rp in range(prof_num):
            random_profile = [self.random_scaled_value(float(envelope[sm]), dist, conf) for sm in self.system_markers]
            random_profiles.append(random_profile)

        rp_df = pd.DataFrame(random_profiles, columns=self.system_markers)

        return rp_df

    def random_scaled_value(self, env_value, dist, conf):
        """

        :param env_value:
        :param dist:
        :param conf:
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
