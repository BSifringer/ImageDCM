import enum
import pandas as pd
import torch
import dask.dataframe as dd
import pickle
import numpy as np
import logging
from omegaconf import OmegaConf, ListConfig
# from pandas_profiling import ProfileReport # Uncomment lines in code to create profile report


LOG = logging.getLogger(__name__)
''' This file contains methods to read from a csv file as well as variable selection enum options '''


def pandas2sindices(cfg, columns=None):
    """Reads a csv, removes category objects and returns tensors based on columns tuple.
    If None given, loads all columns into X. Returns 3 sets of train/val/test
    :param string csv_file: Path to data file.
    :param int chunksize: Give a value to read only Top elements.
    :param  tuple columns: Tuple for X and Z data column list subsets
    :param type lab_col: Name of labels data  (Assumed in X columns)
    :return: torch.Tensor, List of column names (+1)
    """
    debug = cfg.data.loading_debug
    lab_col = cfg.data.label

    df, X_cols, Z_cols = my_load_from_csv(cfg.data, columns=columns)
    df = post_process_df(df, cfg.data, columns)
    ## Project dataframe into Tensor (preprocess beforehand):
    train_idx = np.where(df.subset == 'train')[0]
    val_idx = np.where(df.subset == 'val')[0]
    test_idx = np.where(df.subset == 'test')[0]

    LOG.debug(
        f'Train indices: {train_idx.shape}, val idx: {val_idx.shape}, Test idx: {test_idx.shape}')

    #Make sure to remove all dropped Columns from list
    X_cols.remove('subset')
    X_cols.remove(lab_col)
    if debug:
        df['ResponseID'].astype('category')
        df['ResponseID'] = pd.factorize(df['ResponseID'])[0]
    else:
        X_cols.remove('UserID')
        X_cols.remove('ResponseID')

    return df, [train_idx, val_idx, test_idx], (X_cols, Z_cols)


def pandas2subsets(cfg, columns=None):
    """Reads a csv, removes category objects and returns tensors based on columns tuple.
    If None given, loads all columns into X. Returns 3 sets of train/val/test
    :param string csv_file: Path to data file.
    :param int chunksize: Give a value to read only Top elements.
    :param  tuple columns: Tuple for X and Z data column list subsets
    :param type lab_col: Name of labels data  (Assumed in X columns)
    :return: torch.Tensor, List of column names (+1)
    """
    lab_col = cfg.data.label

    df, [train_idx, val_idx, test_idx], (X_cols, Z_cols) = pandas2sindices(
        cfg, columns)
    X = torch.from_numpy(df[X_cols].values).float()
    Z = torch.from_numpy(df[Z_cols].values.astype('float')).float()
    labels = torch.from_numpy(df[lab_col].values).float()

    del df

    X_train, Z_train, labels_train = X[train_idx], Z[train_idx], labels[train_idx]
    X_val, Z_val, labels_val = X[val_idx], Z[val_idx], labels[val_idx]
    X_test, Z_test, labels_test = X[test_idx], Z[test_idx], labels[test_idx]

    return [[X_train, Z_train, labels_train],
            [X_val, Z_val, labels_val],
            [X_test, Z_test, labels_test]], (X_cols, Z_cols)

    ## For some reason,  this prettier loop version is slower 1.5x for 1'000'000; memory managaement issue?
    # return [[X[idx], Z[idx], labels[idx]] for idx in [train_idx, val_idx, test_idx]]


def pandas2XZY(csv_file, chunksize=None, columns=None, lab_col='Saved', subset=None, dtypes_dict=None):  # Currently Depricated
    """Reads a csv, removes category objects and returns tensors based on columns tuple.
    If None given, loads all columns into X.
    :param string csv_file: Path to data file.
    :param int chunksize: Give a value to read only Top elements.
    :param  tuple columns: Tuple for X and Z data column list subsets
    :param type lab_col: Name of labels data  (Assumed in X columns)
    :return: torch.Tensor, List of column names (+1)
    """

    df, X_cols, Z_cols = my_load_from_csv(
        csv_file, chunksize, columns, dtypes_dict)
    ## Project dataframe into Tensor (preprocess beforehand):
    df['ResponseID'].astype('category')
    df['ResponseID'] = pd.factorize(df['ResponseID'])[0]
    df = df.drop(
        ['ExtendedSessionID'], axis=1).astype('float64')

    if subset is not None:  # if is None, perhaps need to categorize and factorize
        # Return  Train Val or Test only
        df = df.drop(df[df.subset != subset].index, inplace=True)
        df.drop(['subset'], axis=1, inplace=True)

    labels = torch.from_numpy(df[lab_col].values).float()
    #Make sure to remove all dropped Columns from list
    X_cols.remove('ExtendedSessionID')
    X_cols.remove('subset')
    X_cols.remove(lab_col)

    X = torch.from_numpy(df[X_cols].values).float()
    Z = torch.from_numpy(df[Z_cols].values).float()
    return X, Z, labels


def my_load_from_csv(dconf, columns=None):
    """ Load tabular data into a pandas frame
    :param dconf:  data configuartion variables (e.g., cfg.data)
    :param columns:  columns selection to load from file
    :return: loaded dataframe and returns list of loaded columns in case they were not specified
    """
    csv_file = dconf.end_file
    parquet_file = dconf.parquet_file

    if dconf.load_scn1_only:
        csv_file = csv_file[:-4]+ dconf.no_barrier_file_ext + '.csv'
        parquet_file = parquet_file[:-8]+ dconf.no_barrier_file_ext + '.parquet'

    if columns is not None:
        X_cols = columns[0].copy()
        Z_cols = columns[1].copy()
        if isinstance(Z_cols, ListConfig):
            Z_cols = OmegaConf.to_object(Z_cols)
        usecols = list(np.unique(X_cols + Z_cols))

    if dconf.use_parquet:
        # Fast loader with pre-determined data types
        df = pd.read_parquet(parquet_file, engine="fastparquet")
        if dconf.chunksize is not None:
            df = df[:dconf.chunksize]
    else:
        dtypes_dict = None
        if dconf.dtypes_dict_path is not None:
            with open(dconf.dtypes_dict_path, 'rb') as f:
                dtypes_dict = pickle.load(f)
        if dconf.chunksize is not None:
            df_reader = pd.read_csv(
                csv_file, chunksize=dconf.chunksize, usecols=usecols, dtype=dtypes_dict)
            df = next(df_reader)
        else:
            df = pd.read_csv(csv_file, usecols=usecols, dtype=dtypes_dict)
    ## Profiling data:
    # df['Saved'][::2] = df['Saved'][::2]+2 # This is to differntiate in the report the balance of labels (since 2 lines = 1 scenario, saved is 50/50; but we want to know how many times it's pair/odd)
    # profile = ProfileReport(df, title="Pandas Profiling Report")
    # profile.to_file("data_report.html")

    if columns is None:
        X_cols = df.columns
        Z_cols = []
    return df, X_cols, Z_cols


def pandas2tensor(csv_file, chunksize=None):
    """Reads a csv, removes category objects and returns colum names and tensor.
    :param type csv_file: Path to data file.
    :param type chunksize: Give a value to read only Top elements.
    :return: torch.Tensor, List of column names (+1)
    """
    if chunksize is not None:
        df_reader = pd.read_csv(csv_file, chunksize=chunksize)
        df = next(df_reader)
    else:
        df = pd.read_csv(csv_file)
    columns = df.columns
    ## Project dataframe into Tensor (preprocess beforehand):
    df['ResponseID'].astype('category')
    df['ResponseID'] = pd.factorize(df['ResponseID'])[0]
    df = df.drop(
        ['ExtendedSessionID'], axis=1).astype('float64')
    df = torch.from_numpy(df.values).float()  # df is now a torch.Tensor
    return df, columns


def post_process_df(df, dconf, columns):
    ''' Barrier scenarios add unnecessary complexity in the choice alternatives'''
    if dconf.no_barrier_scenario and not dconf.load_scn1_only:
        # remove all lines wher Barrier is 1 grouped by ResponeID
        df = df.groupby('ResponseID').filter(lambda x: x['Barrier'].sum() == 0)
        while len(df)<dconf.chunksize:
            df_2, _,_ = my_load_from_csv(dconf, columns)
            df_2 = df_2.groupby('ResponseID').filter(lambda x: x['Barrier'].sum() == 0)
            df = df.append(df_2)
        df = df[:dconf.chunksize]
    return df
    

#------------------

class UsedAttributes(enum.Enum):
    all = (['ResponseID', 'ExtendedSessionID', 'UserID', 'ScenarioOrder',
            'Intervention', 'PedPed', 'Barrier', 'CrossingSignal',
            'ScenarioTypeStrict', 'DefaultChoice', 'NonDefaultChoice',
            'DefaultChoiceIsOmission', 'NumberOfCharacters',
            'DiffNumberOFCharacters', 'Saved', 'DescriptionShown', 'LeftHand',
            'Oceania', 'SouthAmerica', 'NorthAmerica', 'Europe', 'Africa', 'Asia',
            'UserCountry3', 'Review_age', 'Review_education', 'Review_gender',
            'Review_political', 'Review_religious', 'Man', 'Woman', 'Pregnant',
            'Stroller', 'OldMan', 'OldWoman', 'Boy', 'Girl', 'Homeless',
            'LargeWoman', 'LargeMan', 'Criminal', 'MaleExecutive',
            'FemaleExecutive', 'FemaleAthlete', 'MaleAthlete', 'FemaleDoctor',
            'MaleDoctor', 'Dog', 'Cat',
            'AttributeLevel_Fat',
            'AttributeLevel_Female', 'AttributeLevel_Fit', 'AttributeLevel_High',
            'AttributeLevel_Hoomans', 'AttributeLevel_Less', 'AttributeLevel_Low',
            'AttributeLevel_Male', 'AttributeLevel_More', 'AttributeLevel_Old',
            'AttributeLevel_Pets', 'AttributeLevel_Rand', 'AttributeLevel_Young',
            'Template_Desktop', 'Template_Mobile', 'Review_income_10000',
            'Review_income_15000', 'Review_income_25000', 'Review_income_35000',
            'Review_income_5000', 'Review_income_50000', 'Review_income_80000',
            'Review_income_above100000', 'Review_income_default',
            'Review_income_over10000', 'Review_income_under5000',
            'ScenarioType_Age', 'ScenarioType_Fitness', 'ScenarioType_Gender',
            'ScenarioType_Random', 'ScenarioType_Social', 'ScenarioType_Species',
            'ScenarioType_Utilitarian', 'subset'], [])

    '''
    Removed from bellow options:
    ['ExtendedSessionID', 'PedPed', 'UserCountry3', 'DefaultChoice', 'NonDefaultChoice', 'Review_income_default',
    'ScenarioType_Random', 'ScenarioTypeStrict','ScenarioOrder']
    '''
    #
    # default = (
    #     ['ResponseID', 'UserID',
    #      'Intervention', 'PedPed', 'Barrier', 'CrossingSignal',
    #      'NumberOfCharacters',
    #      'Saved',
    #      'Man', 'Woman', 'Pregnant', 'Stroller', 'OldMan',
    #      'OldWoman', 'Boy', 'Girl', 'Homeless', 'LargeWoman', 'LargeMan',
    #      'Criminal', 'MaleExecutive', 'FemaleExecutive', 'FemaleAthlete',
    #      'MaleAthlete', 'FemaleDoctor', 'MaleDoctor', 'Dog', 'Cat', 'subset'],
    #     []
    #     )
    #
    ### NOTE: NumberOfCharacters creates unstable topology of loss: all STDS of characters explode;
    ### It is however, possible to learn with, and Invert the Hessian matrix by removing it during inversion ....
    ### Explanation? NumChar is 1-1 correlated with the "ensemble" of variables {characters}
    default = (
        ['ResponseID', 'UserID',
         'Saved',
         'Intervention', 'Barrier', 'CrossingSignal',
         # 'NumberOfCharacters',
         'Man', 'Woman', 'Pregnant', 'Stroller', 'OldMan',
         'OldWoman', 'Boy', 'Girl', 'Homeless', 'LargeWoman', 'LargeMan',
         'Criminal', 'MaleExecutive', 'FemaleExecutive', 'FemaleAthlete',
         'MaleAthlete', 'FemaleDoctor', 'MaleDoctor', 'Dog', 'Cat',
         'subset'],

        ['DefaultChoiceIsOmission', 'DiffNumberOFCharacters',
         'AttributeLevel_Fat', 'AttributeLevel_Female', 'AttributeLevel_Fit',
         'AttributeLevel_High', 'AttributeLevel_Hoomans', 'AttributeLevel_Less',
         'AttributeLevel_Low', 'AttributeLevel_Male', 'AttributeLevel_More',
         'AttributeLevel_Old', 'AttributeLevel_Pets', 'AttributeLevel_Rand',
         'AttributeLevel_Young', 'Template_Desktop', 'Template_Mobile',
         'DescriptionShown', 'LeftHand',
         'Oceania', 'SouthAmerica', 'NorthAmerica', 'Europe', 'Africa', 'Asia',
         'Review_age', 'Review_education', 'Review_gender', 'Review_political',
         'Review_religious',
         'Review_income_10000', 'Review_income_15000', 'Review_income_25000',
         'Review_income_35000', 'Review_income_5000', 'Review_income_50000',
         'Review_income_80000', 'Review_income_above100000',
         'Review_income_over10000', 'Review_income_under5000',
         'ScenarioType_Age', 'ScenarioType_Fitness', 'ScenarioType_Gender',
         'ScenarioType_Social', 'ScenarioType_Species',
         'ScenarioType_Utilitarian']
        )
    reduced = (
        ['ResponseID', 'UserID',
         'Saved',
         'Intervention', 'Barrier', 'CrossingSignal',
         # 'NumberOfCharacters',
         'Man', 'Woman', 'Pregnant', 'Stroller', 'OldMan',
         'OldWoman', 'Boy', 'Girl', 'Homeless', 'LargeWoman', 'LargeMan',
         'Criminal', 'MaleExecutive', 'FemaleExecutive', 'FemaleAthlete',
         'MaleAthlete', 'FemaleDoctor', 'MaleDoctor', 'Dog', 'Cat',
         'subset'],

        []
        )
    # Masking the Variable "Man" seems to break others in 100'000 data regime (e.g., OldMan)
    # Removing here to see if Masking brings about same results as Removing
    # Results: python run.py data.varChoice=masked_test is equivalent results to
    # python run.py trainer.tabular_masking=remove trainer.masked_tabular_list=\[Man\]
    masked_test = (
        ['ResponseID', 'UserID',
         'Saved',
         'Intervention', 'Barrier', 'CrossingSignal',
         # 'NumberOfCharacters',
         'Woman', 'Pregnant', 'Stroller', 'OldMan',
         'OldWoman', 'Boy', 'Girl', 'Homeless', 'LargeWoman', 'LargeMan',
         'Criminal', 'MaleExecutive', 'FemaleExecutive', 'FemaleAthlete',
         'MaleAthlete', 'FemaleDoctor', 'MaleDoctor', 'Dog', 'Cat',
         'subset'],

        []
        )
