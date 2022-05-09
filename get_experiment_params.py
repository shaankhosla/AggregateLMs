import pandas as pd

experiment_table_csv_link = "https://docs.google.com/spreadsheets/d/1nVZOPeP8s_zMcnDBw9QRAu_UI3jdbICFpolVc4rwT9A/export?format=csv&id=1nVZOPeP8s_zMcnDBw9QRAu_UI3jdbICFpolVc4rwT9A&gid=818501719"

def get_experiment_configurations(config_number):
    '''
    Read from google sheet outlining experiment and return ensemble configuration details.

    config_number is type int
    '''
    experiment_table = pd.read_csv(experiment_table_csv_link).iloc[:,:7].dropna() ## Ensures only filled in data comes through
    experiment_table = experiment_table[experiment_table['Configuration Num']==config_number]
    model_paths = experiment_table['Path'].to_list()
    num_models_per_type = experiment_table['Number of Models'].astype(int).to_list()
    pruning_factors = experiment_table['Pruning Factor'].to_list()

    return model_paths, num_models_per_type, pruning_factors