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

    ## TODO: Return a dictionary containing the accuracy, f1, precision, and recall scores.
    ## You may use sklearn's precision_recall_fscore_support and accuracy_score methods.

    # precision, recall, f1, support = precision_recall_fscore_support(labels,preds,average='binary')
    # accuracy = accuracy_score(labels,preds)

    accuracy = np.mean(labels==preds)

    precision = np.mean(labels[np.argwhere(preds==1)].flatten())
    recall = np.mean(preds[np.argwhere(labels==1)].flatten())
    f1 = 2 * precision * recall / (precision + recall)

    metrics = {"accuracy":accuracy,
               "precision":precision,
               "recall":recall,
               "f1":f1
              }
    return metrics



def model_init(task_name):
    """Returns an initialized model for use in a Hugging Face Trainer."""
    ## TODO: Return a pretrained RoBERTa model for sequence classification.
    ## See https://huggingface.co/transformers/model_doc/roberta.html#robertaforsequenceclassification.

    task_name_to_kwargs = {
        'CB': {'num_labels': 3},
        'RTE': {'num_labels': 2},
        'BoolQ': {'num_labels': 2},
        'MultiRC': {'num_labels': 2}
    }

    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path="gpt2", **task_name_to_kwargs[task_name])

    tokenizer = GPT2TokenizerFast.from_pretrained(pretrained_model_name_or_path="gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path="gpt2", config=model_config)

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = model.config.eos_token_id

    return model
