import pandas as pd
import numpy as np

experiment_table_csv_link = "https://docs.google.com/spreadsheets/d/1nVZOPeP8s_zMcnDBw9QRAu_UI3jdbICFpolVc4rwT9A/export?format=csv&id=1nVZOPeP8s_zMcnDBw9QRAu_UI3jdbICFpolVc4rwT9A&gid=818501719"

def get_experiment_configurations(config_number, task_name):
    '''
    Read from google sheet outlining experiment and return ensemble configuration details.

    config_number is type int
    task_name is a string. Must be one of [RTE, CB, BoolQ].
    '''
    experiment_table = pd.read_csv(experiment_table_csv_link).iloc[:,:8].dropna() ## Ensures only filled in data comes through
    experiment_table = experiment_table[(experiment_table['Configuration Num'].astype(int)==config_number) & (experiment_table['Task']==task_name)]
    template_model_paths = experiment_table['Path'].to_list()
    num_models_per_type = experiment_table['Number of Models'].astype(int).to_list()
    pruning_factors = experiment_table['Pruning Factor'].to_list()
    is_baseline = experiment_table['Is Baseline'].to_list()
    model_types = experiment_table['Model Type'].to_list()
    #all_model_paths = [[] for _ in range(len(num_models_per_type))]
    all_model_paths = []
    all_pruning_factors = []
    
    for idx, (model_path, num_models, baseline_bool) in enumerate(zip(template_model_paths, num_models_per_type, is_baseline)):
        asterisk_position = model_path.find("*")
        #if "baseline" not in model_path and "checkpoint":
        if not baseline_bool:
            assert asterisk_position != -1
            print(f"Assembling ensemble of bootstrapped models for {task_name}")
            print(f"Ensemble consists of {list(zip(model_types, num_models_per_type))}")
        else:
            print(f"Running baseline model for {task_name}, in path {model_path}")
        model_path_as_list = list(model_path)
        
        if num_models < 10:
            bootstrap_model_nums = np.random.choice(range(0,10), size=num_models, replace=False)
            
            for bootstrap_model_num in bootstrap_model_nums:
                model_path_as_list[asterisk_position] = str(bootstrap_model_num)
                #all_model_paths[idx].append("".join(model_path_as_list))
                all_model_paths.append("".join(model_path_as_list))
        else:
            print("Bootstrap models will be repeating since there has been a request for more than 10")
            num_repeats = num_models // 10
            remainder = num_models % 10
            all_10_bootstraps = []
            
            for i in range(10):
                model_path_as_list[asterisk_position] = str(i)
                all_10_bootstraps.append("".join(model_path_as_list))
            
            for n in range(num_repeats):
                #all_model_paths[idx].extend(all_10_bootstraps)
                all_model_paths.extend(all_10_bootstraps)
                
            bootstrap_model_nums = np.random.choice(range(0,10), size=remainder, replace=False)
            
            for bootstrap_model_num in bootstrap_model_nums:
                model_path_as_list[asterisk_position] = str(bootstrap_model_num)
                #all_model_paths[idx].append("".join(model_path_as_list))
                all_model_paths.append("".join(model_path_as_list))
        
        all_pruning_factors.extend([pruning_factors[idx] for _ in range(num_models)])
        
    return all_model_paths, all_pruning_factors