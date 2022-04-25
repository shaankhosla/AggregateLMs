import numpy as np
from transformers import GPT2ForSequenceClassification, GPT2TokenizerFast, GPT2Config
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    """Computes accuracy, f1, precision, and recall from a 
    transformers.trainer_utils.EvalPrediction object.
    """
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)


    if eval_pred.predictions.shape[1] > 2: # more than two classes
        average_strategy='macro'
    else:
        average_strategy='binary'

    accuracy = np.mean(labels==preds)

    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average=average_strategy)

    metrics = {"accuracy":accuracy,
               "precision":precision,
               "recall":recall,
               "f1":f1}
    return metrics



def model_init(task_name, model_name):
    """Returns an initialized model for use in a Hugging Face Trainer."""
    ## TODO: Return a pretrained RoBERTa model for sequence classification.
    ## See https://huggingface.co/transformers/model_doc/roberta.html#robertaforsequenceclassification.

    task_name_to_kwargs = {
        'CB': {'num_labels': 3},
        'RTE': {'num_labels': 2},
        'BoolQ': {'num_labels': 2},
        'MultiRC': {'num_labels': 2},
        'COPA': {}
    }

    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name, **task_name_to_kwargs[task_name])

    tokenizer = GPT2TokenizerFast.from_pretrained(pretrained_model_name_or_path=model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name, config=model_config)

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = model.config.eos_token_id

    return model
