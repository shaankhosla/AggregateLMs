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
from transformers import RobertaTokenizer, TrainingArguments, Trainer
from transformers import DebertaTokenizer, DebertaModel,DebertaConfig

from datetime import datetime

date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

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
    "-e",
    "--epochs",
    type=int,
    help="Num Epochs",
)

parser.add_argument(
    "-t",
    "--training_batch_size",
    type=int,
    help="Size Training Batch",
)

parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    help="Learning Rate",
)
parser.add_argument(
    "-w",
    "--weight_decay",
    type=float,
    help="Weight Decay",
)
parser.add_argument(
    "-n",
    "--num_models",
    type=int,
    help="Number of models to train",
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
    )
else:
    train_df = data_utils.process_multirc_jsonl(f"{args.data_dir}/train.jsonl", " ")
    val_df, test_df = train_test_split(data_utils.process_multirc_jsonl(f"{args.data_dir}/val.jsonl", " "), test_size=0.5,random_state=42)


#tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
tokenizer = DebertaTokenizer.from_pretrained(pretrained_model_name_or_path="microsoft/deberta-base")

val_data = dataset.CBDataset(val_df, tokenizer, task_name)
test_data = dataset.CBDataset(test_df, tokenizer, task_name)


def main():

    for i in range(args.num_models):
        
        random_shuffle_train_df = train_df.sample(replace=True, frac=1, random_state=i)
        train_data = dataset.CBDataset(random_shuffle_train_df, tokenizer, task_name)
        
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=True,
            do_train=True,
            do_eval=False,
            per_device_train_batch_size=args.training_batch_size,
            num_train_epochs=args.epochs, # due to time/computation constraints
            logging_strategy="no",
            save_strategy="no",
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            disable_tqdm=True)


        model_init_for_task = partial(finetuning_utils.model_init, task_name)
        
        trainer = Trainer(
            args=training_args,
            train_dataset=train_data,
            tokenizer=tokenizer,
            model_init=model_init_for_task,
        )
        
        print("Training model now")
        trainer.train()
        
        save_dir = os.path.join(args.save_dir,'deberta', task_name, date, str(i)) 
        
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        print(f"Done training model, saving model in {save_dir}")

        trainer.save_model(output_dir=save_dir)


if __name__ == "__main__":
    main()
