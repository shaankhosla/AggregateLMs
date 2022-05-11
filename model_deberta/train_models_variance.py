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

from sklearn.model_selection import train_test_split
from transformers import GPT2TokenizerFast, TrainingArguments, Trainer
from transformers import AutoConfig,AutoModelForSequenceClassification,AutoTokenizer

from datetime import datetime
import time
import torch

assert torch.cuda.is_available()
"""
Run instructions:
Variables in the run python command:
{TASK_NAME} (string) - BoolQ, RTE, CB
{MODEL_NAME} (string) - deberta-base, deberta-v3-large,deberta-v3-small, deberta-v3-xsmall

Run command:
python /scratch/${USER}/AggregateLMs/model_deberta/train_models_variance.py -d /scratch/${USER}/superglue/{TASK_NAME}  -o /scratch/${USER}/saved_models/model_checkpoints -s /scratch/${USER}/saved_models/models -m {MODEL_NAME}  -n 10

Hyperparameters are fed from a json file and saved in variable hyperparams. One can just make a dictionary called hyperparams with the keys and values and replicate this behavior.
"""

date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

parser = argparse.ArgumentParser(
    description="Train many models on randomly shuffled data, and save checkpoints"
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
    "-s",
    "--save_dir",
    type=str,
    help="Directory where final model is saved.",
)

parser.add_argument(
    "-n",
    "--num_models",
    type=int,
    help="Number of models to train",
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

f = open("/scratch/ms12768/AggregateLMs/model_deberta/deberta_hyperparams.json")
all_params = json.load(f)

hyperparams = list(filter(lambda x: x['model_name']==args.model_name and x['task']==task_name, all_params))[0]
f.close()

print("Hyperparameters print below")
print(hyperparams)

print("Fine-tuning for task: ", task_name)

if task_name != 'MultiRC':
    train_df = pd.read_json(f"{args.data_dir}/train.jsonl", lines=True, orient="records")
    val_df, test_df = train_test_split(
        pd.read_json(f"{args.data_dir}/val.jsonl", lines=True, orient="records"),
        test_size=0.5,
    )
else:
    train_df = data_utils.process_multirc_jsonl(f"{args.data_dir}/train.jsonl", " ")
    val_df, test_df = train_test_split(data_utils.process_multirc_jsonl(f"{args.data_dir}/val.jsonl", " "), test_size=0.5,random_state=42)


tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token

val_data = dataset.GeneralDataset(val_df, tokenizer, task_name)
test_data = dataset.GeneralDataset(test_df, tokenizer, task_name)



def main():
    start = time.time()
    #print("Training baseline non-shuffled model")
    
    #gradient_accumulation_steps_ = 8
    #gradient_checkpoint_bool = (args.model_name == 'gpt2-large') or (args.model_name == 'gpt2-xl')

    #print(f'Baseline single LM trained time elapsed: {time.time() - start}')

    ## training 10 represents the original bootstrain training data choices
    for i in range(10):

            
        ## Bootstrapped models 
        outer_bootstrap_df = train_df.sample(replace=True, frac=1, random_state=i)
        for j in range(5):
            random_shuffle_train_df = outer_bootstrap_df.sample(replace=True, frac=1, random_state=100+j) ## changing random state 
            
            train_data = dataset.GeneralDataset(random_shuffle_train_df, tokenizer, task_name)
            model_type = 'bootstrapped'

            
            model_id = model_type + "_" + str(i) + "_" + str(j)
            full_output_dir = os.path.join(args.output_dir, 'var_analysis', args.model_name, task_name, model_id)

            if not os.path.isdir(full_output_dir):
                os.makedirs(full_output_dir)

            training_args = TrainingArguments(
                output_dir=full_output_dir,
                overwrite_output_dir=False,
                do_train=True,
                #do_eval=False,
                per_device_train_batch_size=int(hyperparams['train_batch_size']),
                #per_device_eval_batch_size=8,
                num_train_epochs=int(hyperparams['n_epochs']),
                logging_strategy="epoch",
                save_strategy="no",
                evaluation_strategy="no",
                weight_decay=hyperparams['weight_decay'],
                learning_rate=hyperparams['learning_rate'],
                disable_tqdm=True)


            model_init_for_task = partial(finetuning_utils.model_init, task_name, args.model_name)
            
            trainer = Trainer(
                args=training_args,
                train_dataset=train_data,
                #eval_dataset=test_data,
                tokenizer=tokenizer,
                model_init=model_init_for_task,
            )
            
            print("Training model now")
            trainer.train()

            save_dir = os.path.join(args.save_dir, 'var_analysis', args.model_name, task_name, model_id)
            
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            print(f"Done training model, saving model in {save_dir}")

            trainer.save_model(output_dir=save_dir)
    
    print(f'Elapsed time: {time.time() - start}')


if __name__ == "__main__":
    main()