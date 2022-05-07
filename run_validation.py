import argparse
import json
import os
from collections import defaultdict
from functools import partial
from functools import reduce
from lib2to3.pgen2.tokenize import tokenize
import numpy as np

from model_roberta import data_utils
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.nn.utils import prune
from transformers import GPT2ForSequenceClassification
from transformers import GPT2Tokenizer
from transformers import RobertaForSequenceClassification
from transformers import RobertaTokenizer
from transformers import Trainer
from transformers import TrainingArguments
from yaml import load

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


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
        
        
        models.append(model)
        tokenizers.append(tokenizer)
    return models, tokenizers


def get_module_by_name(module, access_string):
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_nonzero(model):
    return sum(torch.count_nonzero(p) for p in model.parameters() if p.requires_grad)


def prune_models(models):
    new_models = []
    prune_factor = 1 - (1 / len(models))
    for model in models:
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
            amount=prune_factor,
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
        votingDict = defaultdict(int)
        summed_probs = []
        for i, model in enumerate(models):
            tokenizer = tokenizers[i]
            tokenizedinput = data_utils.encode_data(
                test_df[i : i + 1], tokenizer, task_name
            )
            input_ids, attention_mask = tokenizedinput
            with torch.inference_mode():
                logits = model(input_ids).logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                summed_probs.append(np.array(probs[0]))

            predicted_class_id = int(torch.argmax(logits, axis=-1)[0])
            votingDict[model.config.id2label[predicted_class_id]] += 1
        y_pred = np.array(summed_probs)
        y_pred = np.sum(summed_probs, axis=0)
        predicted_class_id = int(np.argmax(y_pred, axis=-1))
        y_preds.append(predicted_class_id)

    target_names = [str(x) for x in range(len(logits))]
    print(target_names)
    if len(logits) > 2:  # more than two classes
        average_strategy = "macro"
    else:
        average_strategy = "binary"

    return classification_report(
        y_true, y_preds, target_names=target_names
    ), precision_recall_fscore_support(y_true, y_preds, average=average_strategy)




def main():
    ### these will become args
    TASK = "BoolQ"
#     MODEL_PATHS = ["/scratch/sk8520/AggregateLMs/models/CB/0/", "/scratch/sk8520/AggregateLMs/models/CB/1/"]
    MODEL_PATHS = ['/scratch/ms12768/outputs_deberta_base/outputs_10_BoolQ/saved_models/deberta/BoolQ/2022_05_03-02:47:06_AM/0/', "/scratch/ms12768/outputs_deberta_base/outputs_10_BoolQ/saved_models/deberta/BoolQ/2022_05_03-02:47:06_AM/1/"]
    PRUNE = False
    #### hard code for now
    
    
    
    print("Evaluating", TASK)
    if TASK != "MultiRC":
        test_df = pd.read_json(
            f"./data/{TASK}/val.jsonl", lines=True, orient="records"
        )
    else:
        test_df = data_utils.process_multirc_jsonl(
            f"./data/{TASK}/val.jsonl", " "
        )


    models, tokenizers = load_models(MODEL_PATHS)
    print(len(models), "models")

    if PRUNE:
        models = prune_models(models)
    report, scores = run_evaluation(models, tokenizers, TASK, test_df)
    print(report, scores)



if __name__ == "__main__":
    main()

