import argparse
import json
import os
from collections import defaultdict
from functools import partial
from functools import reduce
from lib2to3.pgen2.tokenize import tokenize

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.nn.utils import prune
from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import GPT2ForSequenceClassification
from transformers import GPT2Tokenizer
from transformers import RobertaForSequenceClassification
from transformers import RobertaTokenizer
from transformers import DebertaV2Tokenizer
from transformers import Trainer
from transformers import TrainingArguments
from yaml import load

from get_experiment_params import get_experiment_configurations
from model_roberta import data_utils
from tqdm import tqdm
import csv
import datetime
import time
experiment_table_csv_link = "https://docs.google.com/spreadsheets/d/1nVZOPeP8s_zMcnDBw9QRAu_UI3jdbICFpolVc4rwT9A/export?format=csv&id=1nVZOPeP8s_zMcnDBw9QRAu_UI3jdbICFpolVc4rwT9A&gid=818501719"




DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device", DEVICE)


def to_device(batch):
    return {k: v.to(DEVICE) for k, v in batch.items()}



def load_models(model_dir):
    models, tokenizers = [], []

    for i, path in enumerate(model_dir):
        print(f"Loading model {i+1} from " + str(path))
        model = AutoModelForSequenceClassification.from_pretrained(
            path, problem_type="multi_label_classification"
        )
        #if "deberta" in model_dir:
        #    tokenizer = DebertaV2Tokenizer.from_pretrained(path)
        #else:
        tokenizer = AutoTokenizer.from_pretrained(path)
        #         tokenizer = RobertaTokenizer.from_pretrained(path)
        
        
        print("type of model: - " + str(type(model)))

        models.append(model.to(DEVICE))
        tokenizers.append(tokenizer)
    return models, tokenizers


def get_module_by_name(module, access_string):
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_nonzero(model):
    return sum(torch.count_nonzero(p) for p in model.parameters() if p.requires_grad)


def prune_models(models, pruning_factors):
    assert len(models) == len(pruning_factors)
    
    print('PRUNING!!')
    new_models = []
    for i, model in enumerate(models):
        print('Count of parameters and nonzero pre prune',count_parameters(model), count_nonzero(model))
        module_tups = []
        for layer in list(model.state_dict().keys()):
            if ".weight" not in layer:
                continue
            x = layer.split(".weight")[0]
            module_tups.append((get_module_by_name(model, x), "weight"))
        prune.global_unstructured(
            parameters=module_tups,
            pruning_method=prune.L1Unstructured,
            amount=pruning_factors[i],
        )

        for module, _ in module_tups:
            prune.remove(module, "weight")
        new_models.append(model)
        print('Count of parameters and nonzero post prune', count_parameters(model), count_nonzero(model))
    print("models pruned!\n")
    return new_models


def run_evaluation_singleLM(model, tokenizer, task_name, test_df):
    y_true = data_utils.extract_labels(test_df, task_name)
    if task_name == 'CB':
        num_cols = 3
    else:
        num_cols = 2
    final_probs = np.zeros((len(y_true), num_cols))

    for i in range(len(y_true)):
        tokenizedinput = data_utils.encode_data(test_df[i : i + 1], 
                                                tokenizer,
                                                task_name)
        input_ids, attention_mask = tokenizedinput
        with torch.inference_mode():
            logits = model(input_ids.to(DEVICE)).logits
            # print("logits below")
            # print(logits)
            # print()
            probs = torch.nn.functional.softmax(logits, dim=1)
            # print("probs below")
            # print(probs)
            # print()
            inter_probs = np.array(probs[0].to('cpu'))
            # print("inter probs")
            # print(inter_probs)
            # print()
            final_probs[i] = inter_probs
    # print("final_probs")
    # print(final_probs)
    # print()


    return final_probs

def run_evaluation(MODEL_PATHS, PRUNING_FACTORS, task_name, test_df):
    y_true = data_utils.extract_labels(test_df, task_name)
    if task_name == 'CB':
        num_cols = 3
    else:
        num_cols = 2
    summed_probs = np.zeros((len(y_true), num_cols))

    for path, pruning_factor in zip(MODEL_PATHS, PRUNING_FACTORS):
        #print("inside load")
        models, tokenizers = load_models([path]) ## will be a list of model size 1
        ## SKIPPING PRUNING FOR NOW. pruning takes in a list of pruning factors. 
        models = prune_models(models, [pruning_factor])
        final_probs = run_evaluation_singleLM(models[0], tokenizers[0], task_name, test_df)
        summed_probs += final_probs
    #print("summed probs")
    #print(summed_probs)
    #print("-----")
    y_preds = np.argmax(summed_probs, axis=1)
    return classification_report(y_true, y_preds, output_dict=True)


# def run_evaluation(models, tokenizers, task_name, test_df):
#     y_preds = []
#     y_true = data_utils.extract_labels(test_df, task_name)

#     for i in range(len(y_true)):
#         summed_probs = []
#         for tokenizer, model in zip(tokenizers, models):
#             tokenizedinput = data_utils.encode_data(
#                 test_df[i : i + 1], tokenizer, task_name
#             )
#             input_ids, attention_mask = tokenizedinput
#             with torch.inference_mode():
#                 logits = model(input_ids.to(DEVICE)).logits
#                 probs = torch.nn.functional.softmax(logits, dim=1)
#                 summed_probs.append(np.array(probs[0].to('cpu')))
            
#         y_pred = np.array(summed_probs)
#         y_pred = np.sum(summed_probs, axis=0)
#         predicted_class_id = int(np.argmax(y_pred, axis=-1))
#         y_preds.append(predicted_class_id)

#     return classification_report(y_true, y_preds, output_dict=True) 


def main(TIME, config_number, TASK):
    MODEL_PATHS, PRUNING_FACTORS = get_experiment_configurations(config_number, TASK)

    f = open(f'validation_report_{TIME}.csv', 'a')
    writer = csv.writer(f)
    
    ###############
    
    print(MODEL_PATHS)
    print(PRUNING_FACTORS)
    
    print("Evaluating", TASK)
    if TASK != "MultiRC":
        test_df = pd.read_json(f"./data/{TASK}/val.jsonl", lines=True, orient="records")
    else:
        test_df = data_utils.process_multirc_jsonl(f"./data/{TASK}/val.jsonl", " ")

    _, test_df = train_test_split(test_df, test_size=0.5, random_state=42)
    
    
    
    #test_df = test_df.sample(5) ###### DELETE #######
    
    #models, tokenizers = load_models(MODEL_PATHS)
    #print(len(models), "models")

 
    #models = prune_models(models, PRUNING_FACTORS)
    
    st_time = time.time()
    report = run_evaluation(MODEL_PATHS, PRUNING_FACTORS, TASK, test_df)
    time_taken = time.time() - st_time
    #print(time_taken, "<--- time taken")
    
    #print('Time for predictions and loading', time_taken)
    accuracy = report['accuracy']
    macro_f1 = report['macro avg']['f1-score']
    
    row = [config_number, TASK, time_taken, accuracy, macro_f1]
    writer.writerow(row)
    f.close()
    

if __name__ == "__main__":
    
    TIME = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    
    f = open(f'validation_report_{TIME}.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['config', 'task', 'time', 'accuracy', 'macro_f1'])
    f.close()
    f = open(f"error_{TIME}.txt", "w")
    f.close()
    
    
    
    experiment_table = pd.read_csv(experiment_table_csv_link).iloc[:,:8].dropna()
    #config_numbers = experiment_table['Configuration Num'].unique().astype(int)

    config_numbers = [8, 50, 51, 52, 53, 54, 110, 111, 112, 114, 115, 116, 117, 16, 17, 20, 21]
    for config_number in tqdm(config_numbers):
        print('Config number', config_number)
        for TASK in ['CB', "BoolQ", "RTE"]:
            print('Task', TASK)
            try:
                main(TIME, config_number, TASK)
            except Exception as e:
                with open(f"error_{TIME}.txt", 'a') as f:
                    f.write(f"ERROR on config {config_number} and task {TASK}!!!!\n")
                    f.write(f"{e}\n\n")
                print('ERROR!!')
                print(e)
                print(config_number)
                print(TASK)
                continue
            
            print('\n\n')
        print('\n\n\n\n')