import torch
import json
import pandas as pd

def process_multirc_jsonl(filepath, separation_token):

    passages_reformatted = {'passage':[], 'question_answer_concat':[], 'label':[]}

    with open(filepath, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)

        passage_text = result['passage']['text']
        question_answers = result['passage']['questions']
        for question_answer in question_answers:
            question_text = question_answer['question']
            answers = question_answer['answers']
            for answer in answers:
                answer_text = answer['text']
                answer_label = answer['label']
                
                passages_reformatted['passage'].append(passage_text)
                passages_reformatted['question_answer_concat'].append(question_text + separation_token + answer_text)
                passages_reformatted['label'].append(answer_label)
  

    processed_df = pd.DataFrame(passages_reformatted)  
    return processed_df

def encode_data(dataset, tokenizer, task_name, max_seq_length=512):
    """Featurizes the dataset into input IDs and attention masks for input into a
     transformer-style model.

     NOTE: This method should featurize the entire dataset simultaneously,
     rather than row-by-row.

  Args:
    dataset: A Pandas dataframe containing the data to be encoded.
    tokenizer: A transformers.PreTrainedTokenizerFast object that is used to
      tokenize the data.
    max_seq_length: Maximum sequence length to either pad or truncate every
      input example to.

  Returns:
    input_ids: A PyTorch.Tensor (with dimensions [len(dataset), max_seq_length])
      containing token IDs for the data.
    attention_mask: A PyTorch.Tensor (with dimensions [len(dataset), max_seq_length])
      containing attention masks for the data.
  """
    ## TODO: Tokenize the questions and passages using both truncation and padding.
    ## Use the tokenizer provided in the argument and see the code comments above for
    ## more details.

    task_to_encode_config = {
      'CB': {'seq1': ['premise'],'seq2':['hypothesis'], 'max_seq_length': 256},
      'RTE': {'seq1': ['premise'],'seq2':['hypothesis'], 'max_seq_length': 256},
      'BoolQ': {'seq1': ['question'],'seq2':['passage'], 'max_seq_length': 256},
      'MultiRC': {'seq1': ['passage'],'seq2':['question_answer_concat'], 'max_seq_length': 256},
      'COPA': {'seq1': ['premise','choice1'],'seq2':['premise','choice2'], 'max_seq_length': 256},
    }

    # Code to concatenate the columns listed in above dict (if there are multiple)

    seq1_parts = [dataset[col] for col in task_to_encode_config[task_name]['seq1']]
    seq2_parts = [dataset[col] for col in task_to_encode_config[task_name]['seq2']]

    seq1 = ""
    seq2 = ""

    for idx, part in enumerate(seq1_parts):
      if idx != 0:
        seq1 = seq1 + " <s> "
      seq1 = seq1 + part
    
    for idx, part in enumerate(seq2_parts):
      if idx != 0:
        seq2 = seq2 + " <s> "
      seq2 = seq2 + part

    # print(seq1[0])
    # print(seq2[0])

    encoded_input = tokenizer(seq1.to_list(), seq2.to_list(), return_tensors='pt', truncation=True, padding='max_length', max_length=task_to_encode_config[task_name]['max_seq_length'], return_attention_mask=True)\

    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']

    return torch.tensor(input_ids), torch.tensor(attention_mask)

def extract_labels(dataset, task_name):
    """Converts labels into numerical labels.

  Args:
    dataset: A Pandas dataframe containing the labels in the column 'label'.

  Returns:
    labels: A list of integers corresponding to the labels for each example,
      where 0 is False and 1 is True.
  """
    ## TODO: Convert the labels to a numeric format and return as a list.

    task_to_labels = {
      'CB': {'neutral':0,'entailment':1,'contradiction':2},
      'RTE': {'entailment':0,'not_entailment':1},
      'BoolQ': {'False':0,'True':1},
      'MultiRC': {},
      'COPA': {}
    }

    return dataset['label'].astype('str').replace(task_to_labels[task_name]).to_numpy().astype(int)