import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scripts.random_profiles import RandomProfileGenerator
from sklearn.preprocessing import Normalizer, StandardScaler

from IPython.core.debugger import set_trace

class ProfileReader:

    def __init__(self, data_file, mechanism_file=None):
        self.df = ''
        self.profile_length = 0
        self.profile_column_name = ''
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

        #try:
        self.df = pd.read_excel(self.data_file)
        self.profile_column_name = self.df.columns.values[0]
        self.sm_columns = self.df.columns.values[1:]  # keep only System:marker.
        self.profile_length = len(self.sm_columns)
        self.name_map = {'agent': 'Agent', 'conc': 'Concentration', 'mech': 'Mechanism',
                         'prof': self.df.columns.values[0]}
        #except FileNotFoundError as err:
        #    raise FileNotFoundError(str(err))

        mech = ''
        if self.mechanism_file is not None:
            mech = pd.read_excel(self.mechanism_file)
        else:
            mech = pd.DataFrame({self.profile_column_name: self.df.iloc[:, 0],
                'Mechanism': ['unknown' for _ in range(len(self.df))]})

        # merge mechanism date into Data frame, adding new column 'Mechanism'
        self.df = pd.merge(left=self.df, right=mech, how='left')

        # add columns for Agent, Concentration
        agents, concs = self.read_agent_conc()
        self.df['Agent'] = agents
        self.df['Concentration'] = concs

        return self.df

    def read_agent_conc(self):
        repeat_pat = re.compile(r'R[0-9]+')
        bsk_pat = re.compile(r'BSK-[A-Z][0-9]+')
        agents = []
        concs = []
        for s in self.df.iloc[:, 0]:
            p_i = s.split(r',')
            conc = p_i[-1].strip()
            p_i = p_i[:-1]  # remove last element
            tok = p_i[-1].strip()  # BSK code or agent or part of agent

            result = bsk_pat.match(tok)  # not interested in BSK code
            if result is not None:
                p_i = p_i[:-1]

            # now we are left with {A} -> Trusted Profile or {E, R, A} -> Profile
            # Agent name can contain commas, too!
            # {A} Exactly one string, Agent name.
            if len(p_i) == 1:
                agent = p_i[0].strip()
            # {E, R, A}, see if middle token has the Regex patters R[0-9]+
            if len(p_i) >= 3:
                rep = p_i[1].strip()
                result = repeat_pat.match(rep)
                if result is not None:  # we found repeat
                    p_i = p_i[2:]  # discard experiment and Repeat and use the remaining list from index 2 up as agent name
                # use the remaining list from index 2 up as agent name
                agent = ','.join(p_i).strip()
            concs.append(conc)
            agents.append(agent)
        return agents, concs

    def get_profile_count(self):
        '''
        How many profiles are in the target?
        :return: int
        '''
        return len(self.df)

    def check_profile_names(self, verbose=True):
        bad_profile_names = []
        pn_format = 'Experiment, R1, Agent Name, BSK-C000001, 30000 nM'
        format_len = len(pn_format.split(','))


        bad_profile_names = [s for s in self.df.iloc[:, 0] if len(s.split(',')) < format_len]
        if len(bad_profile_names) > 0:
            print('Invalid Profile name: required profile name format:', pn_format)




    def get_profile(self, index=None, column_level=1, columns=None):
        '''
        Get the processed DataFrame with the specified indices and columns. Make a copy of self.profile.
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

    def get_profile_names(self):
        """
        :return: The profile names as a Series
        """
        return self.df.iloc[:, 0]

    @staticmethod
    def impute(data, how='group_mean'):
        """
        Impute missing data. Care need to be taken that the data frame does not have str columns values otherwise you get
        empty column values.
        :param data:
        :param how: str 'group_mean'. Calculate the mean value of the missing data point by taking the average readout value within class.
        :return: The profile data with imputed data points.
        """
        grouped = data.groupby(data.index.get_level_values(0).values)

        def f(x):
            return x.fillna(x.mean())
        return grouped.transform(f)



    def plot(self, data=None, agents=None, title='Profile Plots', ylim=None, index=None, figsize=(15, 8),
             legend=True, xticks=True, show=True, **kwds):
        """

        :param data: The data to plot
        :param agents: Optional. Only plot the agents specified.
        :param title: Optional, the title of the plot.
        :param ylim: Optional, y limits of the plot
        :param index: Optional: if supplied and data is not supplied, self.df will be indexed therefore
                        the plot legend will use index.
        :param figsize: plot size
        :param kwds: keywords Options to pass to matplotlib plotting method
        :return:
        """
        if data is None:
            data = self.get_profile(index)

        if agents is not None:
            if isinstance(agents, str):
                agents = [agents]

            if len(agents) > 0:
                q = 'agent in ' + str(agents)
                data = data.query(q)

        ax = data.T.plot(
            figsize=figsize,
            sharey=True,
            subplots=False,
            **kwds);

        if legend:
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);
        else:
            ax.legend_.remove()
        #set_trace()
        x, x_labels, v_line_positions = self.v_line_positions(data.columns.values);

        if xticks:
            #plt.xticks(x, x_labels, rotation='vertical');
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=90)

        plt.ylabel('Log ratio');
        plt.title(title)
        # add vertical lines
        for lp in v_line_positions:
            ax.axvline(x=lp - 0.5)
        if ylim is not None:
            plt.ylim(ylim)
        if show:
            plt.show()
        return ax

    def get_pos_class(self, mech):
        return self.get_profile(index=['mech']).loc[[mech]]

    def get_neg_class(self, data, rpg=None, prof_num=None, dist='rand'):
        if rpg is None:
            rpg = RandomProfileGenerator(envelope_file='data\\SigEnvelopeFile.xml', data_file=data, skip_cols=0)
        if prof_num is None:
            prof_num = len(data)
        neg_class = rpg.get_neg_class(prof_num=prof_num, dist=dist)
        return neg_class

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
        return all_class

    def get_x_y(self, mech, impute='group_mean', normalize_method=None, prof_num=None):
        """
        Convenience method to split data into X and y. Treat one mechanism at a time.
        :param mech: Get data portion for mechanism class
        :param impute: None | 'group_mean'. If not None, impute the positive class using this imputing strategy.
        :param normalize_method: str 'l1', 'l2', 'max', 'ss'. Normalize input profile vectors to unit length.
                            If 'ss', apply standard scaling to each vector, not the feature,
                            See sklearn.preprocessing.Normalizer, sklearn.preprocessing.StandardScaler
        :param prof_num: None|int. If int then the generate this many members of negative class.
        :return: The training data and the trainig labels.
        """
        data = self.get_profile(index=['mech'])
        if impute is not None:
            data = self.impute(data, how=impute)

        pos_class = data.loc[[mech]]
        if normalize_method is not None:
            pos_class = self.normalize(pos_class, normalize_method)

        neg_class = self.get_neg_class(data=pos_class, prof_num=prof_num)
        all_class = pd.concat([pos_class, neg_class]).reset_index(drop=False)

        # encode mechanism to int values
        mech = all_class[['mech']].iloc[0, 0]
        y = all_class['mech'].map({mech: 1, 'neg_class': 0})

        x = all_class.iloc[:, 1:]
        x.set_index(y, inplace=True)

        # if normalize is not None:
        #     scaler = Normalizer(norm=normalize)  # l1, l2, max
        #     cval = x.columns.values
        #     x = pd.DataFrame(scaler.fit_transform(x))
        #     x.columns = cval

        return x, np.array(y)

    @staticmethod
    def normalize(pos_class, method='l2'):
        cval = pos_class.columns.values
        ind = pos_class.index
        if method == 'ss':
            scaler = StandardScaler()
            pos_class = pd.DataFrame(scaler.fit_transform(pos_class.T)).T
        if method in ['l1', 'l2', 'max']:
            scaler = Normalizer(norm=method)  # l1, l2, max
            pos_class = pd.DataFrame(scaler.fit_transform(pos_class))
        pos_class.columns = cval
        pos_class.index = ind
        return pos_class

    # def get_features_labels(self, mech):
    #     """
    #     This method returns is the grand finale of all pre-processing steps combined. It reads in the training profile
    #     data, the mechanism file and the envelop file. It combines the training data with the mechanisms labeling it. It
    #     also generates random profiles and concatenates it with the positive class. Then it separates the features and
    #     labels into two data structures and returns them.
    #
    #     :return:
    #     """
    #     data = self.get_profile(index=['mech'])
    #     pos_class = data.loc[[mech]]
    #
    #     rpg = RandomProfileGenerator(envelope_file='data\\SigEnvelopeFile.xml', data_file=data, skip_cols=0)
    #     neg_class = rpg.get_neg_class(prof_num=len(pos_class), dist='rand')
    #     all_class = self.combine_pos_neg_class(pos_class, neg_class)
    #
    #     return self.get_x_y(all_class)


class TrainedSystemMarkers:
    def __init__(self):
        ''' System-markers trained. The target profiles must have the same system-markers! '''
        self.trained_sm = [
            'BrEPI_IL-1b/TNF-a/IFN-g_24:CD87/uPAR',
            'BrEPI_IL-1b/TNF-a/IFN-g_24:CXCL10/IP-10',
            'BrEPI_IL-1b/TNF-a/IFN-g_24:CXCL9/MIG',
            'BrEPI_IL-1b/TNF-a/IFN-g_24:HLA-DR',
            'BrEPI_IL-1b/TNF-a/IFN-g_24:IL-1alpha',
            'BrEPI_IL-1b/TNF-a/IFN-g_24:MMP-1',
            'BrEPI_IL-1b/TNF-a/IFN-g_24:PAI-I',
            'BrEPI_IL-1b/TNF-a/IFN-g_24:SRB', 'BrEPI_IL-1b/TNF-a/IFN-g_24:tPA',
            'BrEPI_IL-1b/TNF-a/IFN-g_24:uPA',
            'HDFn_IL-1b/TNF-a/IFN-g/EGF/FGF/PDGFbb_24:CD106/VCAM-1',
            'HDFn_IL-1b/TNF-a/IFN-g/EGF/FGF/PDGFbb_24:Collagen III',
            'HDFn_IL-1b/TNF-a/IFN-g/EGF/FGF/PDGFbb_24:CXCL10/IP-10',
            'HDFn_IL-1b/TNF-a/IFN-g/EGF/FGF/PDGFbb_24:CXCL8/IL-8',
            'HDFn_IL-1b/TNF-a/IFN-g/EGF/FGF/PDGFbb_24:CXCL9/MIG',
            'HDFn_IL-1b/TNF-a/IFN-g/EGF/FGF/PDGFbb_24:EGFR',
            'HDFn_IL-1b/TNF-a/IFN-g/EGF/FGF/PDGFbb_24:M-CSF',
            'HDFn_IL-1b/TNF-a/IFN-g/EGF/FGF/PDGFbb_24:MMP-1',
            'HDFn_IL-1b/TNF-a/IFN-g/EGF/FGF/PDGFbb_24:PAI-I',
            'HDFn_IL-1b/TNF-a/IFN-g/EGF/FGF/PDGFbb_24:Proliferation_72hr',
            'HDFn_IL-1b/TNF-a/IFN-g/EGF/FGF/PDGFbb_24:SRB',
            'HDFn_IL-1b/TNF-a/IFN-g/EGF/FGF/PDGFbb_24:TIMP-2',
            'HEK/HDFn_IL-1b/TNF-a/IFN-g/TGF-b_24:CCL2/MCP-1',
            'HEK/HDFn_IL-1b/TNF-a/IFN-g/TGF-b_24:CD54/ICAM-1',
            'HEK/HDFn_IL-1b/TNF-a/IFN-g/TGF-b_24:CXCL10/IP-10',
            'HEK/HDFn_IL-1b/TNF-a/IFN-g/TGF-b_24:IL-1alpha',
            'HEK/HDFn_IL-1b/TNF-a/IFN-g/TGF-b_24:MMP-9',
            'HEK/HDFn_IL-1b/TNF-a/IFN-g/TGF-b_24:SRB',
            'HEK/HDFn_IL-1b/TNF-a/IFN-g/TGF-b_24:TIMP-2',
            'HEK/HDFn_IL-1b/TNF-a/IFN-g/TGF-b_24:uPA',
            'HUVEC/PBMC_LPS_24:CCL2/MCP-1', 'HUVEC/PBMC_LPS_24:CD106/VCAM-1',
            'HUVEC/PBMC_LPS_24:CD142/Tissue Factor', 'HUVEC/PBMC_LPS_24:CD40',
            'HUVEC/PBMC_LPS_24:CD62E/E-Selectin',
            'HUVEC/PBMC_LPS_24:CXCL8/IL-8', 'HUVEC/PBMC_LPS_24:IL-1alpha',
            'HUVEC/PBMC_LPS_24:M-CSF', 'HUVEC/PBMC_LPS_24:sPGE2',
            'HUVEC/PBMC_LPS_24:SRB', 'HUVEC/PBMC_LPS_24:sTNF-alpha',
            'HUVEC/PBMC_SEB/TSST_24:CCL2/MCP-1', 'HUVEC/PBMC_SEB/TSST_24:CD38',
            'HUVEC/PBMC_SEB/TSST_24:CD40',
            'HUVEC/PBMC_SEB/TSST_24:CD62E/E-Selectin',
            'HUVEC/PBMC_SEB/TSST_24:CD69', 'HUVEC/PBMC_SEB/TSST_24:CXCL8/IL-8',
            'HUVEC/PBMC_SEB/TSST_24:CXCL9/MIG',
            'HUVEC/PBMC_SEB/TSST_24:PBMC Cytotoxicity',
            'HUVEC/PBMC_SEB/TSST_24:Proliferation',
            'HUVEC/PBMC_SEB/TSST_24:SRB',
            'HUVEC_IL-1b/TNF-a/IFN-g_24:CCL2/MCP-1',
            'HUVEC_IL-1b/TNF-a/IFN-g_24:CD106/VCAM-1',
            'HUVEC_IL-1b/TNF-a/IFN-g_24:CD141/Thrombomodulin',
            'HUVEC_IL-1b/TNF-a/IFN-g_24:CD142/Tissue Factor',
            'HUVEC_IL-1b/TNF-a/IFN-g_24:CD54/ICAM-1',
            'HUVEC_IL-1b/TNF-a/IFN-g_24:CD62E/E-Selectin',
            'HUVEC_IL-1b/TNF-a/IFN-g_24:CD87/uPAR',
            'HUVEC_IL-1b/TNF-a/IFN-g_24:CXCL8/IL-8',
            'HUVEC_IL-1b/TNF-a/IFN-g_24:CXCL9/MIG',
            'HUVEC_IL-1b/TNF-a/IFN-g_24:HLA-DR',
            'HUVEC_IL-1b/TNF-a/IFN-g_24:Proliferation',
            'HUVEC_IL-1b/TNF-a/IFN-g_24:SRB',
            'HUVEC_IL-4/Histamine_24:CCL2/MCP-1',
            'HUVEC_IL-4/Histamine_24:CCL26/Eotaxin-3',
            'HUVEC_IL-4/Histamine_24:CD106/VCAM-1',
            'HUVEC_IL-4/Histamine_24:CD62P/P-selectin',
            'HUVEC_IL-4/Histamine_24:CD87/uPAR', 'HUVEC_IL-4/Histamine_24:SRB',
            'HUVEC_IL-4/Histamine_24:VEGFR2',
            'CASMC_HCL_IL-1b/TNF-a/IFN-g_24:CCL2/MCP-1',
            'CASMC_HCL_IL-1b/TNF-a/IFN-g_24:CD106/VCAM-1',
            'CASMC_HCL_IL-1b/TNF-a/IFN-g_24:CD141/Thrombomodulin',
            'CASMC_HCL_IL-1b/TNF-a/IFN-g_24:CD142/Tissue Factor',
            'CASMC_HCL_IL-1b/TNF-a/IFN-g_24:CD87/uPAR',
            'CASMC_HCL_IL-1b/TNF-a/IFN-g_24:CXCL8/IL-8',
            'CASMC_HCL_IL-1b/TNF-a/IFN-g_24:CXCL9/MIG',
            'CASMC_HCL_IL-1b/TNF-a/IFN-g_24:HLA-DR',
            'CASMC_HCL_IL-1b/TNF-a/IFN-g_24:IL-6',
            'CASMC_HCL_IL-1b/TNF-a/IFN-g_24:LDLR',
            'CASMC_HCL_IL-1b/TNF-a/IFN-g_24:M-CSF',
            'CASMC_HCL_IL-1b/TNF-a/IFN-g_24:Proliferation',
            'CASMC_HCL_IL-1b/TNF-a/IFN-g_24:Serum Amyloid A',
            'CASMC_HCL_IL-1b/TNF-a/IFN-g_24:SRB'
        ]

    def __call__(self):
        return self.trained_sm


class TargetProcessor(ProfileReader):
    """
    Convenience class to read and validate target excel file. The target file must have the same number of Syste:Markers
    as the the trained set.
    """

    def __init__(self, data_file, verbose=True, remove_nulls=True):
        """
        Initialize the target profiles and make sure we dont have missing columns or extra columns.
        :param data_file: The target profile data we want to predict. Call the super init method to initialize. Then
        remove the extra columns and reorder the profile in the same order as the trained model.
        """
        super().__init__(data_file, mechanism_file=None)
        if remove_nulls:
            prof_mis = self.df[self.df.isnull().any(1)].iloc[:, 0]
            print("\n\nProfiles with missing values will not be analyzed:\n\n", '; '.join(prof_mis))
            self.df = self.df[~self.df.isnull().any(1)]


        # Note: The model was trained with data from 'SMC_IL-1b/TNF-a/IFN-g_24' system.
        # In the TOXCast data set the this was replaced with 'CASMC_HCL_IL-1b/TNF-a/IFN-g_24' in feature names and
        # can be used in place of 'SMC_IL-1b/TNF-a/IFN-g_24'.

        # rename SMC to CASMC
        renamed = {s: s.replace('SMC_IL-1b/TNF-a/IFN-g_24:', 'CASMC_HCL_IL-1b/TNF-a/IFN-g_24:') for s in
                   self.df.columns.values if s.startswith('SMC_IL-1b/TNF-a/IFN-g_24')}
        self.df.rename(columns=renamed, inplace=True)
        self.sm_columns = self.df.columns.values[1:-3]  # keep only System:marker.

        # if missing columns, delete data, print message
        cols_not_in_target = list(set(TrainedSystemMarkers()()) - set(self.sm_columns))
        if len(cols_not_in_target) > 0:
            print('Target file missing system:markers:', cols_not_in_target)
            self.df.drop(self.df.index, inplace=True)
            return

        if verbose:
            print("Removing extra target colunms:", list(set(self.sm_columns) - set(TrainedSystemMarkers()())))

        # remove those system-markers that were not trained and rearrange
        self.df = self.df[[self.profile_column_name, *TrainedSystemMarkers()(), 'Mechanism', 'Agent', 'Concentration']]

        self.sm_columns = self.df.columns.values[1:-3]  # keep only System:marker.
        self.profile_length = len(self.sm_columns)

    def check_missing_sm(self):
        """
        Make sure we have the right system:readouts. We cant do SVM with missing data.
        :return: A list of System:readouts missing from the target data file.
        """
        return list(set(TrainedSystemMarkers()()) - set(self.sm_columns))

    def profile_row(self):
        """
        Iterate over target data one row at a time.
        :return: str, Series. The profile name and the data as a Series.
        """
        data = self.get_profile(index=['prof'])
        for profile, row in data.iterrows():
            yield profile, row

    def get_target(self):
        """
        :return: DataFrame the data portion of the target profiles, profile as index.
        """
        return self.get_profile(index=['prof'])

