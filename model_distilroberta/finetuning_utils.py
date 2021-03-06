import numpy as np
from transformers import AutoModelForSequenceClassification
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



def model_init(task_name):
    """Returns an initialized model for use in a Hugging Face Trainer."""
    ## TODO: Return a pretrained DistilRoBERTa model for sequence classification.
    ## See https://huggingface.co/transformers/model_doc/roberta.html#robertaforsequenceclassification.

    task_name_to_kwargs = {
        'CB': {'num_labels': 3},
        'RTE': {},
        'BoolQ': {},
        'MultiRC': {},
        'COPA': {}
    }


    model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", **task_name_to_kwargs[task_name])

    return model
