## Choosing to import numpy to make calculations easier
import numpy as np
from transformers import GPT2ForSequenceClassification, GPT2TokenizerFast, GPT2Config

def compute_metrics(eval_pred):
    """Computes accuracy, f1, precision, and recall from a 
    transformers.trainer_utils.EvalPrediction object.
    """
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    ## TODO: Return a dictionary containing the accuracy, f1, precision, and recall scores.
    ## You may use sklearn's precision_recall_fscore_support and accuracy_score methods.

    ## labels and preds are numpy array objects according to huggingface's docs
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



def model_init():
    """Returns an initialized model for use in a Hugging Face Trainer."""
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path="gpt2", num_labels=2)

    tokenizer = GPT2TokenizerFast.from_pretrained(pretrained_model_name_or_path="gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path="gpt2", config=model_config)

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = model.config.eos_token_id

    #model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=2)

    return model
