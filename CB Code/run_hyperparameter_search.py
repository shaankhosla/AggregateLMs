"""Run a hyperparameter search on a RoBERTa model fine-tuned on CB.

Example usage:
    python run_hyperparameter_search.py CB/
"""
import argparse
import CB
import data_utils
import finetuning_utils
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast, TrainingArguments, Trainer
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray import tune
import ray

ray.init(num_gpus=1)

parser = argparse.ArgumentParser(
    description="Run a hyperparameter search for finetuning a RoBERTa model on the CB dataset."
)
parser.add_argument(
    "data_dir",
    type=str,
    help="Directory containing the CB dataset. Can be downloaded from SuperGLUE resources",
)

args = parser.parse_args()

# Since the labels for the test set have not been released, we will use half of the
# validation dataset as our test dataset for the purposes of this assignment.
train_df = pd.read_json(f"{args.data_dir}/train.jsonl", lines=True, orient="records")
val_df, test_df = train_test_split(
    pd.read_json(f"{args.data_dir}/val.jsonl", lines=True, orient="records"),
    test_size=0.5,
)

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
train_data = CB.CBDataset(train_df, tokenizer)
val_data = CB.CBDataset(val_df, tokenizer)
test_data = CB.CBDataset(test_df, tokenizer)

## TODO: Initialize a transformers.TrainingArguments object here for use in
## training and tuning the model. Consult the assignment handout for some
## sample hyperparameter values.

## TODO: Initialize a transformers.Trainer object and run a Bayesian
## hyperparameter search for at least 5 trials (but not too many) on the 
## learning rate. Hint: use the model_init() and
## compute_metrics() methods from finetuning_utils.py as arguments to
## Trainer(). Use the hp_space parameter in hyperparameter_search() to specify
## your hyperparameter search space. (Note that this parameter takes a function
## as its value.)
## Also print out the run ID, objective value,
## and hyperparameters of your best run.

### NEED TO LOOK AT THIS AND NOT SURE WHERE LOGGING AND MODEL CHECKPOINTS AND ALL OF THAT GO TO BE HONEST.
training_args = TrainingArguments(
    #output_dir="/scratch/pfi203/hw3/outputs3",
    #output_dir="/home/pranab/Natural Language Understanding/HW 3/outputs",
    output_dir="outputs",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    per_gpu_train_batch_size=8,
    per_gpu_eval_batch_size=64,
    num_train_epochs=3, # due to time/computation constraints
    logging_steps=500,
    logging_first_step=True,
    save_steps=1000,
    evaluation_strategy = "epoch", # evaluate at the end of every epoch
    weight_decay=0.01,
    disable_tqdm=True
)

### LOOKED AT THESE ARGUMENTS AND IT LOOKED GOOD: I DID put in the compute metrics, model init, and put in the right training and val data
trainer = Trainer(
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    model_init=finetuning_utils.model_init,
    compute_metrics=finetuning_utils.compute_metrics
)

    
    # Choose among schedulers:
    # https://docs.ray.io/en/latest/tune/api_docs/schedulers.html
    #scheduler=ASHAScheduler(metric="objective", mode="max"))
def hp_space_call(trial):
    
    return {"learning_rate": tune.uniform(1e-5, 5e-5)}#, 3e-5, 4e-5, 5e-5]}

def main():
    #trainer.train() ## i think im supposed to skip this
    print("hyperparam search model")
    best_trial = trainer.hyperparameter_search(
    hp_space=hp_space_call,
    direction="maximize",
    backend="ray",
    search_alg=BayesOptSearch(mode="max"),
    n_trials=3,
    compute_objective=lambda x: x['eval_accuracy']
)
    print("After hyperparam search, best run below:")
    print(best_trial)

if __name__ == "__main__":
    main()
