import pandas as pd
# https://github.com/martinblech/xmltodict
# conda install -c conda-forge xmltodict
import xmltodict as x2d
from numpy.random import rand


class RandomProfileGenerator:
    # indexes in xml to select profile type/confidence
    SCREENING = 0
    TRUSTED = 1
    PROFILE = 2
    _99 = 0
    _95 = 1

    def __init__(self, envelope_file='SigEnvelopeFile.xml', data_file='SVMData.xlsx'):
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

    def get_random_profiles(self, prof_num=10, envelope=SCREENING, conf=_95):
        value_list = self.envelope_dict['java']['object']['void'][envelope]['object'][1]['void'][conf]['object']['void']
        # populate values from value_list into a dict, "system:marker':value
        envelope = {}
        for item in value_list:
            envelope[item['string']] = item['float']

        #####################################
        # now generate random weak profiles #
        #####################################

        ## Generate random Profiles
        random_profiles = []
        for rp in range(prof_num):
            random_profile = [float(envelope[sm]) * (rand() * 2 - 1) for sm in self.system_markers]
            random_profiles.append(random_profile)

        rp_df = pd.DataFrame(random_profiles, columns=self.system_markers)

        return rp_df
