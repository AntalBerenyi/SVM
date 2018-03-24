import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Notebooks.scripts.random_profiles import RandomProfileGenerator

class ProfileReader:

    def __init__(self, data_file, mechanism_file=None):
        self.df = ''
        self.profile_length = 0
        self.name_map = {}
        self.data_file = data_file
        self.mechanism_file = mechanism_file
        self.sm_columns = []
        self.read_excel_()
        # this is a map of what the user specifies to the actual column name in the data frame


    def read_excel_(self):
        '''
        Parse in the Profile Excel file. Index the rows into MultiIndex, containg 'agent', 'conc' and 'profile'
        Columns are parsed as System:Marker, they can be requested later as MultiIndex.
        Data is internally represented as a DataFrame, the first column is the profile name, the last column in the
        mechanism.
        :return: The indexed data frame with Profile data.
        '''

        try:
            self.df = pd.read_excel(self.data_file)
            self.sm_columns = self.df.columns.values[1:]  # keep only System:marker.
            self.profile_length = len(self.sm_columns)
            self.name_map = {'agent': 'Agent', 'conc': 'Concentration', 'mech': 'Mechanism',
                             'prof': self.df.columns.values[0]}
        except Exception as err:
            return format(err)

        mech = ''
        if self.mechanism_file is not None:
            mech = pd.read_excel(self.mechanism_file)
        else:
            mech = pd.DataFrame({'Mechanism': ['unknown' for _ in range(len(self.df))]})

        # merge mechanism date into Data frame, adding new column 'Mechanism'
        self.df = pd.merge(left=self.df, right=mech, how='left')
        # add columns for Agent, Concentration
        self.df['Agent'] = [s.split(',')[2].strip() for s in self.df.iloc[:, 0]]
        self.df['Concentration'] = [s.split(',')[4].strip() for s in self.df.iloc[:, 0]]

        return self.df

    def get_profile(self, index=None, column_level=1, columns=None):
        '''
        Read Excel file containing BioMAP profile.
        :param index: str or array like. What part of the profile to use as index. For example 'profile' or ['agent', 'conc', 'mech']
                    'profile will use the entire profile name as index'
        :param column_level: int 1|2 Use one or multi-index. If column_level = 1, the column names are System:marker.
                    If column_level = 2, column index is ('System', 'Marker')
        :param columns: Generated columns that can are derived from the profile name. It can be a list like ['agent', 'mech', 'prof'] or a string like 'agent'
        :return: A DataFrame with data loaded from an Excel file.
        '''

        data2 = self.df.copy()
        if index is not None:

            # turn columns into indices
            if isinstance(index, str):
                index = [index]

            ind_col_names = [self.name_map[s] for s in index]
            index2 = []
            for idx in ind_col_names:
                index2.append([s for s in self.df[idx]])
            tuples = list(zip(*index2))
            if len(index2) == 1:
                data2.index = index2[0]
                data2.index.rename(index[0], inplace=True)
            else:
                data2.index = pd.MultiIndex.from_tuples(tuples, names=index)


        #################################
        # keep these columns in dataframe
        if columns is None:
            columns = []

        if isinstance(columns, str):
            columns = [columns]

        keep_columns = set(columns)
        all_columns = set(self.name_map.keys())

        # drop these columns
        drop_columns = list(all_columns - keep_columns)
        drop_columns = [self.name_map[s] for s in drop_columns]

        if len(drop_columns) > 0:
            data2 = data2.drop(drop_columns, inplace=False, axis=1)

        ###### multiindex if columns are system-markers ########
        if (column_level == 2) and (self.profile_length == len(data2.columns)):
            col_index = np.transpose([s.split(':') for s in data2.columns.values])
            data2.columns = pd.MultiIndex.from_tuples(list(zip(*col_index)), names=['System', 'Marker'])

        return data2


    def v_line_positions(self, col_names=None):
        '''

        :param col_names: list or None. If none use the leaded profile sys-markers. If not None, use the supplied list.
        :return:
        '''

        if col_names is None:
            col_names = self.sm_columns

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

    def get_system_markers(self, agg=False):
        """
        Get the System markers as a dta frame for convenience
        :return: DataFrame with columns 'System', 'Marker'
        """
        sm = pd.DataFrame([s.split(':') for s in self.sm_columns])
        q = sm.groupby([0, 1]).count()
        q.index.rename(['System', 'Marker'], inplace=True)
        q.reset_index(level=[0, 1], inplace=True)
        if agg:
            q = q.groupby(['System']).agg(lambda x: ', '.join(list(x))).reset_index()
        return q

    def get_system_marker_count(self):
        """
        Get the System markers as a dta frame for convenience
        :return: DataFrame with columns 'System', 'Marker'
        """
        q = self.get_system_markers().groupby('System', as_index=False).count()
        return q

    def get_mechanism_count(self):
        '''

        :return: A DataFrame with mechanism-counts
        '''
        data = self.df
        return pd.DataFrame(data.groupby(data.Mechanism).size(), columns=['Count']).reset_index().\
            sort_values(by='Count', ascending=False).reset_index(drop=True)

    def get_agent_count(self):
        '''

        :return: A DataFrame with agent-counts
        '''
        data = self.df
        return pd.DataFrame(data.groupby(data.Agent).size(), columns=['Count']).reset_index().\
            sort_values(by='Count', ascending=False).reset_index(drop=True)

    def get_mechanisms(self):
        return list(self.df['Mechanism'].unique())

    def get_agents(self):
        return list(self.df['Agent'].unique())

    @staticmethod
    def impute(data, how='group_mean'):
        """
        Impute missing data. Care need to be taken that the data frame does not have str columns values otherwise you get
        empty column values.
        :param data:
        :param how:
        :return:
        """
        grouped = data.groupby(data.index.get_level_values(0).values)

        def f(x):
            return x.fillna(x.mean())
        return grouped.transform(f)

    def plot(self, data=None, agents=None, title='Profile Plots', ylim=None):
        """

        :param data: The data to plot
        :param agents: Optional. Only plot the agents specified.
        :param title: Optional, the title of the plot.
        :return:
        """
        if data is None:
            data = self.get_profile()

        if agents is not None:
            if isinstance(agents, str):
                agents = [agents]

            if len(agents) > 0:
                q = 'agent in ' + str(agents)
                data = data.query(q)

        ax = data.T.plot(
            figsize=(15, 8),
            sharey=True,
            subplots=False);

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);
        x, x_labels, v_line_positions = self.v_line_positions(data.columns.values);
        plt.xticks(x, x_labels, rotation='vertical');
        plt.ylabel('Log ratio');
        plt.suptitle(title, fontsize=16)
        # add vertical lines
        for lp in v_line_positions:
            plt.axvline(x=lp - 0.5)
        if ylim is not None:
            plt.ylim(ylim)
        plt.show()
        return ax

    @staticmethod
    def combine_pos_neg_class(pos_class, neg_class):
        """
        Combine the positive and negative classes
        :param pos_class: A data frame with a single mechanism as index values
        :param neg_class: A data frame with 'neg_class' as index values
        :return: A combined DataFrame with positive class on top and negative class on bottom.
        """
        # Generate all classes by combining positive and negative classes
        all_class = pd.concat([pos_class, neg_class]).reset_index(drop=False)
        # encode mechanism to int values
        mech = all_class[['mech']].iloc[0,0]
        all_class['mech'] = all_class['mech'].map({mech: 1, 'neg_class': 0})
        return all_class

    @staticmethod
    def get_x_y(all_class):
        x = all_class.iloc[:, 1:]
        y = all_class.iloc[:, 0]
        return x, y

    def get_features_labels(self, mech):
        """
        This method returns is the grand finale of all pre-processing steps combined. It reads in the training profile
        data, the mechanism file and the envelop file. It combines the training data with the mechanisms labeling it. It
        also generates random profiles and concatenates it with the positive class. Then it separates the features and
        labels into two data structures and returns them.

        :return:
        """
        data = self.get_profile(index=['mech'])
        pos_class = data.loc[[mech]]

        rpg = RandomProfileGenerator(envelope_file='data\\SigEnvelopeFile.xml', data_file=data, skip_cols=0)
        neg_class = rpg.get_neg_class(prof_num=len(pos_class), dist='rand')
        all_class = self.combine_pos_neg_class(pos_class, neg_class)

        return self.get_x_y(all_class)

