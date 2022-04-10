from transformers import TrainingArguments

from run_glue import train


def main():
    t = {
        "learning_rate": 0.000045,
        "seed": 42,
        "output_dir": "./",
        "do_eval": True,
        "do_predict": False,
        "do_train": True,
        "overwrite_output_dir": True,
    }
    training_arguments = TrainingArguments(**t)
    ModelArguments = {
        "model_name_or_path": "microsoft/deberta-v3-small",
        "config_name": None,
        "tokenizer_name": None,
        "cache_dir": None,
        "use_fast_tokenizer": True,
        "model_revision": "main",
        "use_auth_token": False,
    }
    DataTrainingArguments = {
        "task_name": "mrpc",
        "dataset_name": None,
        "dataset_config_name": None,
        "max_seq_length": 256,
        "overwrite_cache": False,
        "pad_to_max_length": True,
        "max_train_samples": None,
        "max_eval_samples": None,
        "max_predict_samples": None,
        "train_file": None,
        "validation_file": None,
        "test_file": None,
    }
    train(ModelArguments, DataTrainingArguments, training_arguments)


if __name__ == "__main__":
    main()
