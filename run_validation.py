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
from transformers import Trainer
from transformers import TrainingArguments
from yaml import load

from get_experiment_params import get_experiment_configurations
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

    for i in tqdm(range(len(y_true))):
        votingDict = defaultdict(int)
        summed_probs = []
        for i, model in enumerate(models):
            tokenizer = tokenizers[i]
            tokenizedinput = data_utils.encode_data(
                test_df[i : i + 1], tokenizer, task_name
            )
            input_ids, attention_mask = tokenizedinput
            with torch.inference_mode():
                logits = model(input_ids.to(DEVICE)).logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                summed_probs.append(np.array(probs[0].to('cpu')))

            predicted_class_id = int(torch.argmax(logits, axis=-1)[0])
            votingDict[model.config.id2label[predicted_class_id]] += 1
        y_pred = np.array(summed_probs)
        y_pred = np.sum(summed_probs, axis=0)
        predicted_class_id = int(np.argmax(y_pred, axis=-1))
        y_preds.append(predicted_class_id)

        
    if task_name == 'CB':  # more than two classes
        average_strategy = "macro"
    else:
        average_strategy = "binary"

    return classification_report(y_true, y_preds, output_dict=True) 


def main():
    ### these will become args
    TASK = "BoolQ"
    PRUNE = False
    config_number = 17
    MODEL_PATHS, PRUNING_FACTORS = get_experiment_configurations(config_number, TASK)
    

    f = open('validation_report.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['config', 'accuracy', 'macro_f1'])
    ###############
    
    print(MODEL_PATHS)

    
    print("Evaluating", TASK)
    if TASK != "MultiRC":
        test_df = pd.read_json(f"./data/{TASK}/val.jsonl", lines=True, orient="records")
    else:
        test_df = data_utils.process_multirc_jsonl(f"./data/{TASK}/val.jsonl", " ")

    _, test_df = train_test_split(test_df, test_size=0.5, random_state=42)
    
    models, tokenizers = load_models(MODEL_PATHS)
    print(len(models), "models")

    if PRUNE:
        models = prune_models(models, PRUNING_FACTORS)

    report = run_evaluation(models, tokenizers, TASK, test_df)
    accuracy = report['accuracy']
    macro_f1 = report['macro avg']['f1-score']
    
    row = [config_number, accuracy, macro_f1]
    writer.writerow(row)
    f.close()
    

if __name__ == "__main__":
    main()
