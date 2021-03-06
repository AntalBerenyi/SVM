import pandas as pd
import numpy as np


class ProfileReader:

    def __init__(self, data_file, mechanism_file=None):
        self.data_file = data_file
        self.mechanism_file = mechanism_file

    def parse_profiles(self, keep_column_names=True, drop_profiles=True):
        '''
        :param keep_column_names: If False, return the DataFrame with columns set to MultiIndex. With the default False
                value the column names are like System:Readout.
        :param drop_profiles: If True, drop the column that has the profile name.
        :return: A DataFrame with data loaded from an Excel file.
        '''
        data = ''

        try:
            data = pd.read_excel(self.data_file)
        except Exception as err:
            return format(err)

        mech = ''
        if self.mechanism_file is not None:
            mech = pd.read_excel(self.mechanism_file)
        else:
            mech = pd.DataFrame({'Mechanism': ['unknown' for _ in range(len(data))]})

        # Make index from (mechanims, agent, concentration)
        index = [[s for s in mech["Mechanism"]],
                 [s.split(',')[2] for s in data.iloc[:, 0]],
                 [s.split(',')[4] for s in data.iloc[:, 0]]]

        tuples = list(zip(*index))
        # Drop profile column
        if drop_profiles:
            data.drop([data.columns.values[0]], inplace=True, axis=1)
        data.index = pd.MultiIndex.from_tuples(tuples, names=['mech', 'agent', 'conc'])

        col_index = np.transpose([s.split(':') for s in data.columns.values])
        if not keep_column_names:
            data.columns = pd.MultiIndex.from_tuples(list(zip(*col_index)), names=['System', 'Marker'])
        return data



    def v_line_positions(self, col_names=None):
        '''

        :param col_names: list or None. If none use the leaded profile sys-markers. If not None, use the supplied list.
        :return:
        '''

        if col_names is None:
            data = pd.read_excel(self.data_file)
            col_names = data.columns.values[1:]

        last_sys = ""
        x = [i for i in range(len(col_names))]
        x_labels = []
        v_line_positions = []
        for i, sm in enumerate(col_names):
            s, m = sm.split(':')
            if (last_sys != s):
                x_labels.append("{}:{}".format(s, m))
                last_sys = s
                v_line_positions.append(i)
            else:
                x_labels.append(m)

        v_line_positions.append(len(x))
        return x, x_labels, v_line_positions

