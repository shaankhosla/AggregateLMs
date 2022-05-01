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
from transformers import GPT2TokenizerFast, TrainingArguments, Trainer
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray import tune
import ray

ray.init()

parser = argparse.ArgumentParser(
    description="Run a hyperparameter search for finetuning a RoBERTa model on the CB dataset."
)
parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    help="Directory containing the CB dataset. Can be downloaded from SuperGLUE resources",
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
full_output_dir = f"{args.output_dir}/{args.model_name}/{task_name}"
gradient_checkpoint_bool = (args.model_name == 'gpt2-large') or (args.model_name == 'gpt2-xl') ## only gradient checkpoint if large model


print("Fine-tuning for task: ", task_name, args.model_name)

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

tokenizer = GPT2TokenizerFast.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token
train_data = dataset.GeneralDataset(train_df, tokenizer, task_name)
val_data = dataset.GeneralDataset(val_df, tokenizer, task_name)
test_data = dataset.GeneralDataset(test_df, tokenizer, task_name)

def hp_space_call(trial):
    
    return {"learning_rate": tune.uniform(1e-5, 5e-5),
            "weight_decay": tune.uniform(5e-3, 2e-1)}

def main():
    start = time.time()
    gradient_accumulation_steps_ = 8 ## Thus we change batch size to [8, 16, 32] / 4 to simulate batch size 32
    
    ## lets just test and see if batch size with large epochs runs and in what time for various models 
    for num_train_epochs_ in [3,5,4,7]:
        for train_batch_size in [1,2,4]:
            print(f"Before hpsearch run: num train epochs: {num_train_epochs_}. train batch size: {train_batch_size}. effective train batch size: {train_batch_size*gradient_accumulation_steps_}")
            training_args = TrainingArguments(
                output_dir=full_output_dir,
                overwrite_output_dir=False,
                do_train=True,
                do_eval=True,
                per_device_train_batch_size=train_batch_size,
                per_device_eval_batch_size=2,
                num_train_epochs=num_train_epochs_, # due to time/computation constraints
                logging_steps=500,
                logging_first_step=True,
                save_strategy="no",
                evaluation_strategy = "epoch",# evaluate at the end of every epoch
                gradient_accumulation_steps=gradient_accumulation_steps_,
                gradient_checkpointing=gradient_checkpoint_bool,
                disable_tqdm=True)


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
            n_trials=6,
            compute_objective=lambda x: x[eval_method_dict[task_name]]
            )
            print("After hyperparam search, best run below:")
            print(best_trial)
            print(f"num train epochs: {num_train_epochs_}. train batch size: {train_batch_size}. effective train batch size: {train_batch_size*gradient_accumulation_steps_}")
            print("----------------------------------------------------------------------------------")
            print("----------------------------------------------------------------------------------")
            print("----------------------------------------------------------------------------------")
            print(f'hyperparameter search complete, elapsed: {time.time() - start}')

if __name__ == "__main__":
    main()
