"""Run a hyperparameter search on a RoBERTa model fine-tuned on a SuperGLUE task.

"""
import argparse
import dataset
import data_utils
import finetuning_utils
import json
import pandas as pd
import os
from functools import partial
import time

from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaTokenizerFast, TrainingArguments, Trainer
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray import tune
import ray

ray.init() #num_gpus=2 - technically don't need to specify here according to documentation

parser = argparse.ArgumentParser(
    description="Run a hyperparameter search for finetuning a RoBERTa model on the CB dataset."
)
parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    help="Directory containing the relevant SuperGLUE dataset.",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="Directory containing the relevant SuperGLUE dataset.",
)

parser.add_argument(
    "-m",
    "--model_name",
    type=str,
    help="Model name / path that huggingface will recognize in the from_pretrained() function",
)

args = parser.parse_args()

# Since the labels for the test set have not been released, we will use half of the
# validation dataset as our test dataset for the purposes of this assignment.
print("Data is in: ", args.data_dir)


task_name = os.path.basename(args.data_dir)


print("Fine-tuning for task: ", task_name)

if task_name != 'MultiRC':
    train_df = pd.read_json(f"{args.data_dir}/train.jsonl", lines=True, orient="records")
    val_df, test_df = train_test_split(
        pd.read_json(f"{args.data_dir}/val.jsonl", lines=True, orient="records"),
        test_size=0.5,
        random_state=42
    )
else:
    train_df = data_utils.process_multirc_jsonl(f"{args.data_dir}/train.jsonl", " ")
    val_df, test_df = train_test_split(data_utils.process_multirc_jsonl(f"{args.data_dir}/val.jsonl", " "), test_size=0.5,random_state=42)


tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name)
train_data = dataset.CBDataset(train_df, tokenizer, task_name)
val_data = dataset.CBDataset(val_df, tokenizer, task_name)
test_data = dataset.CBDataset(test_df, tokenizer, task_name)

# print(train_data[0])
# print(train_data[1])

"""training_args = TrainingArguments(
    output_dir="/scratch/pfi203/outputs/model_checkpoints",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    #per_gpu_train_batch_size=8,
    per_gpu_eval_batch_size=64,
    #num_train_epochs=3, # due to time/computation constraints
    logging_steps=500,
    logging_first_step=True,
    save_strategy="epoch",
    evaluation_strategy = "epoch", # evaluate at the end of every epoch
    #weight_decay=0.01,
    disable_tqdm=True
)


model_init_for_task = partial(finetuning_utils.model_init, task_name)

### LOOKED AT THESE ARGUMENTS AND IT LOOKED GOOD: I DID put in the compute metrics, model init, and put in the right training and val data
trainer = Trainer(
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    model_init=model_init_for_task,
    compute_metrics=finetuning_utils.compute_metrics
)"""

    
    # Choose among schedulers:
    # https://docs.ray.io/en/latest/tune/api_docs/schedulers.html
    #scheduler=ASHAScheduler(metric="objective", mode="max"))
def hp_space_call(trial):
    
    return {"learning_rate": tune.uniform(1e-5, 5e-5),
            "weight_decay": tune.uniform(5e-3, 2e-1)}

def main():
    start = time.time()

    for num_train_epochs_ in [3,5]: #removing 7 temporarily bc 3/5 failed
        for train_batch_size in [32, 16]:
            training_args = TrainingArguments(
                output_dir=args.output_dir,
                overwrite_output_dir=True,
                do_train=True,
                do_eval=True,
                per_gpu_train_batch_size=train_batch_size,
                per_gpu_eval_batch_size=64,
                num_train_epochs=num_train_epochs_, # due to time/computation constraints
                logging_steps=500,
                logging_first_step=True,
                save_strategy="epoch",
                evaluation_strategy = "epoch", # evaluate at the end of every epoch
                #weight_decay=0.01,
                disable_tqdm=False) #disable when evaluating how long job will take


            model_init_for_task = partial(finetuning_utils.model_init, task_name, args.model_name)
            eval_method_dict = {'CB':"eval_f1",
                                'MultiRC':'eval_f1',
                                'BoolQ':'eval_accuracy',
                                'RTE':'eval_accuracy',
                                'COPA':'eval_accuracy'}
            
            trainer = Trainer(
                args=training_args,
                train_dataset=train_data,
                eval_dataset=val_data,
                tokenizer=tokenizer,
                model_init=model_init_for_task,
                compute_metrics=finetuning_utils.compute_metrics
            )
            print("hyperparam search model")
            best_trial = trainer.hyperparameter_search(
            hp_space=hp_space_call,
            direction="maximize",
            backend="ray",
            search_alg=BayesOptSearch(mode="max"),
            n_trials=5,
            compute_objective=lambda x: x[eval_method_dict[task_name]]
            )
            print("After hyperparam search, best run below:")
            print(best_trial)
            print(f"num train epochs: {num_train_epochs_}. train batch  size: {train_batch_size}")
            print("----------------------------------------------------------------------------------")
            print("----------------------------------------------------------------------------------")
            print("----------------------------------------------------------------------------------")
            print(f'hyperparameter search complete, elapsed: {time.time() - start}')

if __name__ == "__main__":
    main()
