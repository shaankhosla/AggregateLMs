"""Run a hyperparameter search on a RoBERTa model fine-tuned on a SuperGLUE task.

"""
import argparse
import json
import os
from datetime import datetime
from functools import partial

import data_utils
import dataset
import finetuning_utils
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from transformers import Trainer, Seq2SeqTrainer
from transformers import TrainingArguments
from dataclasses import dataclass, field
import torch


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

if task_name != "MultiRC":
    train_df = pd.read_json(
        f"{args.data_dir}/train.jsonl", lines=True, orient="records"
    )
    val_df, test_df = train_test_split(
        pd.read_json(f"{args.data_dir}/val.jsonl", lines=True, orient="records"),
        test_size=0.5,
    )
else:
    train_df = data_utils.process_multirc_jsonl(f"{args.data_dir}/train.jsonl", " ")
    val_df, test_df = train_test_split(
        data_utils.process_multirc_jsonl(f"{args.data_dir}/val.jsonl", " "),
        test_size=0.5,
        random_state=42,
    )


tokenizer = T5Tokenizer.from_pretrained("t5-small")

val_data = dataset.CBDataset(val_df, tokenizer, task_name)
test_data = dataset.CBDataset(test_df, tokenizer, task_name)




# @dataclass
# class T2TDataCollator():
#     def __call__(self, batch):
#         """
#         Take a list of samples from a Dataset and collate them into a batch.
#         Returns:
#             A dictionary of tensors
#         """
#         input_ids = torch.stack([example['input_ids'] for example in batch])
#         lm_labels = torch.stack([example['target_ids'] for example in batch])
#         lm_labels[lm_labels[:, :] == 0] = -100
#         attention_mask = torch.stack([example['attention_mask'] for example in batch])
#         decoder_attention_mask = torch.stack([example['target_attention_mask'] for example in batch])
        

#         return {
#             'input_ids': input_ids, 
#             'attention_mask': attention_mask,
#             'labels': lm_labels, 
#             'decoder_attention_mask': decoder_attention_mask
#         }



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
            num_train_epochs=args.epochs, 
            logging_strategy="no",
            save_strategy="no",
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            disable_tqdm=True,
        )

        model_init_for_task = partial(finetuning_utils.model_init, task_name)

        trainer = Seq2SeqTrainer(
            args=training_args,
            train_dataset=train_data,
            tokenizer=tokenizer,
            model_init=model_init_for_task,
        )

        print("Training model now")
        trainer.train()

        save_dir = os.path.join(args.save_dir, "roberta", task_name, date, str(i))

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        print(f"Done training model, saving model in {save_dir}")

        trainer.save_model(output_dir=save_dir)


if __name__ == "__main__":
    main()
