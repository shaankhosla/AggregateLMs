import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from transformers import RobertaForSequenceClassification
from transformers import T5ForConditionalGeneration
from transformers import T5Model
from transformers import T5Tokenizer
from data_utils import decode


tokenizer = T5Tokenizer.from_pretrained("t5-small")

def compute_metrics(eval_pred):
    """Computes accuracy, f1, precision, and recall from a
    transformers.trainer_utils.EvalPrediction object.
    """
    print('Eval pred')
    print(eval_pred)
    preds = eval_pred[0]
    print(preds)
    answers = [decode(tokenizer, x) for x in preds]
    print(answer)
    import sys;sys.exit()
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    ## TODO: Return a dictionary containing the accuracy, f1, precision, and recall scores.
    ## You may use sklearn's precision_recall_fscore_support and accuracy_score methods.

    # precision, recall, f1, support = precision_recall_fscore_support(labels,preds,average='binary')
    # accuracy = accuracy_score(labels,preds)

    if eval_pred.predictions.shape[1] > 2:  # more than two classes
        average_strategy = "macro"
    else:
        average_strategy = "binary"

    accuracy = np.mean(labels == preds)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=average_strategy
    )

    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    return metrics


def model_init(task_name):
    """Returns an initialized model for use in a Hugging Face Trainer."""
    ## TODO: Return a pretrained RoBERTa model for sequence classification.
    ## See https://huggingface.co/transformers/model_doc/roberta.html#robertaforsequenceclassification.

    task_name_to_kwargs = {
        "CB": {"num_labels": 3},
        "RTE": {},
        "BoolQ": {},
        "MultiRC": {},
        "COPA": {},
    }

#     model = T5Model.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    return model
