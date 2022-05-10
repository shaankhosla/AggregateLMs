import pandas as pd
import numpy as np

experiment_table_csv_link = "https://docs.google.com/spreadsheets/d/1nVZOPeP8s_zMcnDBw9QRAu_UI3jdbICFpolVc4rwT9A/export?format=csv&id=1nVZOPeP8s_zMcnDBw9QRAu_UI3jdbICFpolVc4rwT9A&gid=818501719"
suffix_map = {"distilgpt2":{'BoolQ':list("2022_05_10-01_44_54_AM/"),
                            'CB':list("2022_05_10-01_48_55_AM/"), 
                            'RTE':list("2022_05_10-01_41_57_AM/")},
              "gpt2":{'BoolQ':list("2022_05_10-01_35_25_AM/"),
                      'CB':list("2022_05_10-01_35_51_AM/"),
                      'RTE':list("2022_05_10-01_35_25_AM/")}}
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
    
    all_model_paths = []
    all_pruning_factors = []
    print(f"Ensemble consists of {list(zip(model_types, num_models_per_type))}\n")
    
    for idx, (model_path, num_models, baseline_bool, model_type) in enumerate(zip(template_model_paths, num_models_per_type, is_baseline, model_types)):
        asterisk_position = model_path.find("*")
        #if "baseline" not in model_path and "checkpoint":
        
        if not int(baseline_bool):
            assert asterisk_position != -1
            print(f"Assembling ensemble of bootstrapped models for {task_name}, in modified path {model_path}\n")
        else:
            print(f"Running baseline model for {task_name}, in path {model_path}\n")
        model_path_as_list = list(model_path)
        
        bootstrap_model_nums = np.random.choice(range(0, max(10, num_models)), size=num_models, replace=False)

        for bootstrap_model_num in bootstrap_model_nums:
            
            model_path_as_list[asterisk_position] = str(bootstrap_model_num)
            
            ## gpt2 and distilgpt2 have extra models in different paths. Manual fix.
            if bootstrap_model_num >= 10:
                print("Manually fixing suffixes for gpt2 and distilgpt2 in specific experiments")
                model_path_as_list_modified = model_path_as_list[:-23]
                model_path_as_list_modified.extend(suffix_map[model_type][task_name])
                all_model_paths.append("".join(model_path_as_list_modified))
            else:
                all_model_paths.append("".join(model_path_as_list))
        
        all_pruning_factors.extend([pruning_factors[idx] for _ in range(num_models)])
        
    return all_model_paths, all_pruning_factors