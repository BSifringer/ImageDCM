import tarfile
import pandas as pd
import pickle

import hydra
import submitit
from os import path
import numpy as np
from numpy.random import choice

from omegaconf import OmegaConf, DictConfig, open_dict


'''
This File contains all operations done to the original MIT data csv files (Survey and SharedResponses)
It creates a new csv file which will be used and processed based on the different experiment's  dataloaders
With 16Go RAM, 2.7 Ghz i7 quad-core, (i.e., good laptop) the processing takes a few minutes.
'''


def merge_full2survey(cfg) -> None:
    ''' Loads survey data to memory, then chunksize load and merges only entries
    from SharedResponses that have the same UserID. A new csv is created -
    '''
    chunksize = cfg.data.merge_chunksize

    df = pd.read_csv(cfg.data.survey_file,  low_memory=False)
    df_new_list = []

    with tarfile.open(cfg.data.full_file, "r:*") as tar:
        csv_path = tar.getnames()[0]
        for df_tomerge in pd.read_csv(tar.extractfile(csv_path), low_memory=False, chunksize=chunksize):
            df_chunk = df.merge(df_tomerge, on=['ResponseID', 'ExtendedSessionID', 'UserID', 'ScenarioOrder',
                                                'Intervention', 'PedPed', 'Barrier', 'CrossingSignal', 'AttributeLevel',
                                                'ScenarioTypeStrict', 'ScenarioType', 'DefaultChoice',
                                                'NonDefaultChoice', 'DefaultChoiceIsOmission', 'NumberOfCharacters',
                                                'DiffNumberOFCharacters', 'Saved', 'Template', 'DescriptionShown',
                                                'LeftHand', 'UserCountry3'])

            df_new_list.append(df_chunk)

    df_new = pd.concat(df_new_list)
    df_new.to_csv(cfg.data.merged_file, index=False)
    return 0


def sort_preprocess_data(cfg):
    ''' All data preprocessing on the merged csv file. A new final file is created '''
    df = pd.read_csv(cfg.data.merged_file)
    print('Merged File Loaded')
    ## This solution is very slow, use new one (share on StackOverflow? With Merge solutions)
    # df = df.groupby('UserID')
    # df = df.filter(lambda x: len(x) < 97)  # Remove Users with too many answers
    # #
    # df = df.groupby('ResponseID')
    # # Remove single line ResponseIDs (need 2 lines for binary choice)
    # df = df.filter(lambda x: len(x) > 1)
    ## Remove Users that answered too often (some number seem like bots)
    df_keepUsers = df.groupby(['UserID'])['ResponseID'].count()
    keep_usersDict = dict(
        zip((df_keepUsers < 97).keys(), (df_keepUsers < 97).values))
    df['keep_users'] = df['UserID'].map(keep_usersDict)
    df.drop(df[df['keep_users'] == 0].index, axis=0, inplace=True)
    print('Done filtering Users')
    ## Remove singe ResponseIDs (need 2 for scenario frames):
    # Simulate that 2 responseIDs are count == 1
    df_dropRes = df.groupby(['ResponseID'])['UserID'].count()
    remove_RespDict = dict(
        zip((df_dropRes < 2).keys(), (df_dropRes < 2).values))
    df['remove_resp'] = df['ResponseID'].map(remove_RespDict)
    df.drop(df[df['remove_resp'] == 1].index, axis=0, inplace=True)

    df.drop(['remove_resp', 'keep_users'], axis=1, inplace=True)
    print('Done filtering unusable ResponseIDs (<1)')

    # NaNs mainly by review age or "Random" Scenario type (=> DefaultChoices are NAN)
    # df['DefaultChoice'].fillna(0, inplace=True)
    # df['NonDefaultChoice'].fillna(0, inplace=True)
    df['DefaultChoiceIsOmission'].fillna(0, inplace=True)
    # df.drop(df[df['Review_age'].isna()].index, axis=0)
    # Change all inf values to NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.drop(df[df.isna().any(axis=1)].index, axis=0, inplace=True)
    # df.dropna(axis=0, how="any", inplace=True)


    if not cfg.data.barrier_keep:
        print('Filtering out Scnearios 2 and 3')
        # remove when sum of barrier per responseID is not 0
        df_barrier = df.groupby(['ResponseID'])['Barrier'].sum()
        df_barrier_dropped = df_barrier[df_barrier != 0]
        df_barrier_dropped = df_barrier_dropped.index
        df.drop(df[df['ResponseID'].isin(df_barrier_dropped)].index,    axis=0, inplace=True)

    print('Done filtering')


    ValueReplace = {
        'CrossingSignal': {2: -1},
        'Review_education': {'underHigh': 1/6, 'high': 2/6, 'vocational': 3/6,
                             'college': 4/6, 'bachelor': 5/6, 'graduate': 1,
                             'default': 0, 'others': 0},
        'Review_gender': {'male': 1, 'female': -1, 'others': 0,
                          'default': 0, 'apache helicopter': 0, 'nan': 0}
                     }

    df.replace(ValueReplace, inplace=True)
    # df['DescriptionShown'].fillna(0)
    # df['LeftHand'].fillna(0)

    ValueReplace = {'ScenarioType': {'Social Status': 'Social'}}
    df.replace(ValueReplace, inplace=True)
    print('Replaced Values Done')


    print('Reducing Overall Data Memory Usage (printing sample)')
    reduce_memory_usage(df, cfg.data.dtypes_dict_path)

    #----------------------------UserCountry3-----------------------------------#
    add_continents(df)
    print('Adding Continents Done')


    df = pd.get_dummies(
        df, columns=['AttributeLevel', 'Template', 'Review_income', 'ScenarioType'])
    print('Variables to categorical done')

    '''
    #-----------------------------AttibuteLevel---------------------------------#
    df = pd.get_dummies(df, columns=['AttributeLevel'])
    print('Atribute Level to categorical done')
    #-----------------------------Template---------------------------------#
    df = pd.get_dummies(df, columns=['Template'])
    print('Template to categorical done')

    #-----------------------------Review_income---------------------------------#
    df = pd.get_dummies(df, columns=['Review_income'])
    print('Review_income to categorical done')
    #-------------------------------Scenario-----------------------------------#
    # for i in range(len(df)):
    #     if df['ScenarioType'][i]=='Social Status':
    #        df['ScenarioType'][i]= 'Social'
    ValueReplace = {'ScenarioType': {'Social Status': 'Social'}}
    df.replace(ValueReplace, inplace=True)
    df = pd.get_dummies(df, columns=['ScenarioType'])
    print('ScenarioType to categorical done')
    ###########################################################scenariType
    '''

    # Official website accepts numbers betwwen 18 and 75
    df.loc[(df["Review_age"] > 75) | (df["Review_age"] < 18), "Review_age"] = 0
    # Scale Age by 100 for input
    df["Review_age"] = df["Review_age"]/100.
    # Remove strange valued answers :
    df.drop(df[df["Review_religious"] < 0].index, inplace=True)
    print('Fixed spurious data, Scaled age')

    # del df['UserCountry3']
    # del df['DefaultChoice']
    # del df['NonDefaultChoice']
    # del df['Review_income_default']
    # del df['ScenarioType_Random']
    # del df['ScenarioTypeStrict']

    # Share this solution somewhere (mainly that map is quick)
    ids = df['UserID'].unique()
    val_percent = 1 - cfg.data.train_perc-cfg.data.test_perc
    subsets = choice(['train', 'val', 'test'], ids.shape, p=[
                     cfg.data.train_perc, val_percent, cfg.data.test_perc])
    map_dict = dict(zip(ids, subsets))
    df['subset'] = df['UserID'].map(map_dict)
    print('Adding susbsets complete')

    df.sort_values(by=['ResponseID', 'Intervention'], inplace=True)
    if cfg.data.barrier_keep:
        # df.to_csv(cfg.data.end_file, index=False)
        pass # dirty hack avoiding overwriting
    else:
        df.to_csv(cfg.data.end_file[:-4]+cfg.data.no_barrier_file_ext+'.csv', index=False)
    print('Final Dataframe has # rows: {}'.format(len(df)))
    if cfg.data.save_parquet_file:
        if cfg.data.barrier_keep:
            # df.to_parquet(cfg.data.parquet_file)
            pass
        else:
            df.to_parquet(cfg.data.parquet_file[:-8]+cfg.data.no_barrier_file_ext+'.parquet')
    return 0


def reduce_memory_usage(df, dict_path):
    # print(df[:500000].info(memory_usage='deep'))
    float_list = df.columns[df.dtypes == 'float64']
    float_list = float_list.drop('UserID')

    object_list = df.columns[df.dtypes == 'object']
    object_list = object_list.drop('ResponseID')

    df[df.columns[df.dtypes == 'int64']
       ] = df[df.columns[df.dtypes == 'int64']].astype('int8')
    df[float_list] = df[float_list].astype('int8')
    df[object_list] = df[object_list].astype('category')
    df[['ExtendedSessionID']] = df[['ExtendedSessionID']].astype('float64')
    df[['Review_age']] = df[['Review_age']].astype('float64')
    df[['Review_education']] = df[['Review_education']].astype('float64')

    # print(df[:500000].info(memory_usage='deep'))
    ### Save the dtypes dict for faster loading
    dtypes_dict = df.dtypes.to_dict()
    file_path = dict_path
    with open(file_path, 'wb') as f:
        pickle.dump(dtypes_dict, f)
    return 0


def add_continents(df):
    df.insert(20, "Asia", 0)
    df.insert(20, "Africa", 0)
    df.insert(20, "Europe", 0)
    df.insert(20, "NorthAmerica", 0)
    df.insert(20, "SouthAmerica", 0)
    df.insert(20, "Oceania", 0)

    Africa = ['DZA', 'AGO', 'BWA', 'BDI', 'CMR', 'CPV', 'CAF', 'TCD', 'COM', 'MYT', 'COG',
              'COD', 'BEN', 'GNQ', 'ETH', 'ERI', 'DJI', 'GAB', 'GMB', 'GHA', 'GIN', 'CIV',
              'KEN', 'LSO', 'LBR', 'LBY', 'MDG', 'MWI', 'MLI', 'MRT', 'MUS', 'MAR', 'MOZ',
              'NAM', 'NER', 'NGA', 'GNB', 'REU', 'RWA', 'SHN', 'STP', 'SEN', 'SYC', 'SLE',
              'SOM', 'ZAF', 'ZWE', 'SSD', 'ESH', 'SDN', 'SWZ', 'TGO', 'TUN', 'UGA', 'EGY',
              'TZA', 'BFA', 'ZMB']

    Asia = ['AFG', 'AZE', 'BHR', 'BGD', 'ARM', 'BTN', 'IOT', 'BRN', 'MMR', 'KHM', 'LKA', 'CHN',
            'TWN', 'CXR', 'CCK', 'CYP', 'GEO', 'PSE', 'HKG', 'IND', 'IDN', 'IRN', 'IRQ', 'ISR',
            'JPN', 'KAZ', 'JOR', 'PRK', 'KOR', 'KWT', 'KGZ', 'LAO', 'LBN', 'MAC', 'MYS', 'MDV',
            'MNG', 'OMN', 'NPL', 'PAK', 'PHL', 'TLS', 'QAT', 'RUS', 'SAU', 'SGP', 'VNM', 'SYR',
            'TJK', 'THA', 'ARE', 'TUR', 'TKM', 'UZB', 'YEM']

    Europe = ['ALB', 'AND', 'AZE', 'AUT', 'ARM', 'BEL', 'BIH', 'BGR', 'BLR', 'HRV', 'CYP',
              'CZE', 'DNK', 'EST', 'FRO', 'FIN', 'ALA', 'FRA', 'GEO', 'DEU', 'GIB', 'GRC',
              'VAT', 'HUN', 'ISL', 'IRL', 'ITA', 'KAZ', 'LVA', 'LIE', 'LTU', 'LUX', 'MLT',
              'MCO', 'MDA', 'MNE', 'NLD', 'NOR', 'POL', 'PRT', 'ROU', 'RUS', 'SMR', 'SRB',
              'SVK', 'SVN', 'ESP', 'SJM', 'SWE', 'CHE', 'TUR', 'UKR', 'MKD', 'GBR', 'GGY',
              'JEY', 'IMN']

    NorthAmerica = ['ATG', 'BHS', 'BRB', 'BMU', 'BLZ', 'VGB', 'CAN', 'CYM', 'CRI', 'CUB',
                    'DMA', 'DOM', 'SLV', 'GRL', 'GRD', 'GLP', 'GTM', 'HTI', 'HND', 'JAM',
                    'MTQ', 'MEX', 'MSR', 'ANT', 'CUW', 'ABW', 'SXM', 'BES', 'NIC', 'UMI',
                    'PAN', 'PRI', 'BLM', 'KNA', 'AIA', 'LCA', 'MAF', 'SPM', 'VCT', 'TTO',
                    'TCA', 'USA', 'VIR']

    Oceania = ['ASM', 'AUS', 'SLB', 'COK', 'FJI', 'PYF', 'KIR', 'GUM', 'NRU', 'NCL', 'VUT',
               'NZL', 'NIU', 'NFK', 'MNP', 'UMI', 'FSM', 'MHL', 'PLW', 'PNG', 'PCN', 'TKL',
               'TON', 'TUV', 'WLF', 'WSM']

    SouthAmerica = ['ARG', 'BOL', 'BRA', 'CHL', 'COL', 'ECU', 'FLK', 'GUF', 'GUY', 'PRY',
                    'PER', 'SUR', 'URY', 'VEN']

    # df['UserCountry3'].fillna(0, inplace=True)
    continents = {'Africa': Africa, 'Asia': Asia, 'Europe': Europe,
                  'NorthAmerica': NorthAmerica, 'Oceania': Oceania, 'SouthAmerica': SouthAmerica}
    country_to_continent = {country: continent for continent,
                            co_list in continents.items() for country in co_list}
    country_to_continent.update({0: 'UserCountry3'})
    keys_df_country = np.array(
        [country_to_continent[country] for country in df['UserCountry3'].values])
    # print(keys_df_country)

    for continent in continents.keys():
        df[continent] = keys_df_country == continent


@ hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> int:
    # file exist or preprocess all; call functions until last one
    print('------- Data Preprocessing ----------')
    if not path.exists(cfg.data.merged_file) or cfg.data.merge_files:
        print('Merging Full data and Survey Data')
        merge_full2survey(cfg)
        print('Merging Done')

    if not path.exists(cfg.data.end_file) or cfg.data.preprocess_data:
        print('Preprocessing and Sorting Merged Data')
        sort_preprocess_data(cfg)
    print('Done')


if __name__ == "__main__":
    main()
