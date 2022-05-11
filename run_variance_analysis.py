import argparse
import json
import os
from collections import defaultdict
from functools import partial
from functools import reduce
from lib2to3.pgen2.tokenize import tokenize
from pprint import pprint
from re import M
import datetime

import numpy as np
import pandas as pd
import torch
import random
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
from transformers import Trainer
from transformers import TrainingArguments
from yaml import load

#from get_experiment_params import get_experiment_configurations
from model_roberta import data_utils
from tqdm import tqdm
import csv



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

    new_models = []
    for i, model in enumerate(models):
        print(count_parameters(model), count_nonzero(model))
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
        print(count_parameters(model), count_nonzero(model))
    print("models pruned!")
    return new_models



def run_evaluation(models, tokenizers, task_name, test_df):
    y_preds = []
    y_true = data_utils.extract_labels(test_df, task_name)

    for i in range(len(y_true)):
        summed_probs = []
        for tokenizer, model in zip(tokenizers, models):
            tokenizedinput = data_utils.encode_data(
                test_df[i : i + 1], tokenizer, task_name
            )
            input_ids, attention_mask = tokenizedinput
            with torch.inference_mode():
                logits = model(input_ids.to(DEVICE)).logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                summed_probs.append(np.array(probs[0].to('cpu')))
            
        y_pred = np.array(summed_probs)
        y_pred = np.sum(summed_probs, axis=0)
        predicted_class_id = int(np.argmax(y_pred, axis=-1))
        y_preds.append(predicted_class_id)
    print(y_preds)

    return classification_report(y_true, y_preds, output_dict=True) 

def main():
    TIME = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filePath = 'variance_validation_report_'+str(TIME)+'.csv'
    
    f = open(filePath, 'w')
    writer = csv.writer(f)
    writer.writerow(['config','model name', 'task','single_accuracy','single_macro_f1','double_bootstrapped_accuracy', 'double_bootstrapped_macro_f1'])

    ### these will become args
    #Sample path -> /scratch/ms12768/outputs_deberta_xsmall/baseline_CB/deberta/CB/2022_05_06-01:54:24_AM/0/
    variance_analysis_csv_link = "https://docs.google.com/spreadsheets/d/1nVZOPeP8s_zMcnDBw9QRAu_UI3jdbICFpolVc4rwT9A/export?gid=1501288471&format=csv"
    experiment_table = pd.read_csv(variance_analysis_csv_link).iloc[:,:8].dropna() ## Ensures only filled in data comes through
    print("table",experiment_table)
    #TASK = "BoolQ"
    PRUNE = False
    config_number = 1
    CASES = ["Single","Double"]
    #MODEL_PATHS, PRUNING_FACTORS = get_experiment_configurations(config_number, TASK)
    PRUNING_FACTORS = 0
    #Samplebasepath = "/scratch/ms12768/double_bootstrapped_models/deberta-base/BoolQ/var_analysis/microsoft/deberta-base/BoolQ/"
    for index,row in experiment_table.iterrows():
        TASK = row['Task']
        MODEL_NAME = row['Model Type']
        doublebasepath = row['Double Bootstrap Path']
        singlebasepath = row['Single Bootstrap Path']
        config_number = row['Configuration Num']
        single_macro_f1 = 0
        single_accuracy = 0
        for i in range(10):
            for CASE in CASES:
                if CASE =="Single":
                    MODEL_PATHS = []
                    if "pfi203" in singlebasepath:    
                        MODEL_PATHS.append(singlebasepath.replace("*",str(i))+"/")
                    else:
                        MODEL_PATHS.append(singlebasepath.replace("*",str(i))+"/")

                    #writer.writerow(['config', 'accuracy', 'macro_f1'])
                    #####    
                    print(MODEL_PATHS)
                    print("Evaluating", TASK)
                    if TASK != "MultiRC":
                        test_df = pd.read_json(f"./data/super/{TASK}/val.jsonl", lines=True, orient="records")
                    else:
                        test_df = data_utils.process_multirc_jsonl(f"./data/super/{TASK}/val.jsonl", " ")

                    _, test_df = train_test_split(test_df, test_size=0.5, random_state=42)
                    
                    models, tokenizers = load_models(MODEL_PATHS)
                    print(len(models), "models")

                    if PRUNE:
                        models = prune_models(models, PRUNING_FACTORS)

                    report = run_evaluation(models, tokenizers, TASK, test_df)
                    single_accuracy = report['accuracy']
                    single_macro_f1 = report['macro avg']['f1-score']


                if CASE =="Double":
                    MODEL_PATHS = []
                    for j in range(5):
                        MODEL_PATHS.append(doublebasepath+"/bootstrapped_"+str(i)+"_"+str(j)+"/")
                    #writer.writerow(['config', 'accuracy', 'macro_f1'])
                    #####    
                    print(MODEL_PATHS)
                    print("Evaluating", TASK)
                    if TASK != "MultiRC":
                        test_df = pd.read_json(f"./data/super/{TASK}/val.jsonl", lines=True, orient="records")
                    else:
                        test_df = data_utils.process_multirc_jsonl(f"./data/super/{TASK}/val.jsonl", " ")
                    _, test_df = train_test_split(test_df, test_size=0.5, random_state=42)
                    
                    models, tokenizers = load_models(MODEL_PATHS)
                    print(len(models), "models")

                    if PRUNE:
                        models = prune_models(models, PRUNING_FACTORS)

                    report = run_evaluation(models, tokenizers, TASK, test_df)
                    double_accuracy = report['accuracy']
                    double_macro_f1 = report['macro avg']['f1-score']
                    row_to_write = [config_number,MODEL_NAME,TASK, single_accuracy, single_macro_f1,double_accuracy,double_macro_f1]
                    writer.writerow(row_to_write)
    f.close()
if __name__ == "__main__":
    main()