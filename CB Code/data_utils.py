import torch


def encode_data(dataset, tokenizer, max_seq_length=512):
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
    zipped = zip(dataset["premise"], dataset["hypothesis"])
    mapped = list(map(lambda x: tokenizer(text=x[0], text_pair=x[1],
                                          truncation="only_first", # only truncate the premise (this might cause problems cuz last sentence of premise is important?)
                                          max_length=max_seq_length,
                                          padding="max_length"),
                      zipped))
    input_ids, attention_mask = list(zip(*[(x['input_ids'], x['attention_mask']) for x in mapped]))

    return torch.tensor(input_ids), torch.tensor(attention_mask)

def extract_labels(dataset):
    """Converts labels into numerical labels.

  Args:
    dataset: A Pandas dataframe containing the labels in the column 'label'.

  Returns:
    labels: A list of integers corresponding to the labels for each example,
      where 0 is False and 1 is True.
  """
    ## TODO: Convert the labels to a numeric format and return as a list.
    def label_mapper(label):
      if label == 'neutral':
          return 0
      if label == 'entailment':
          return 1
      if label == 'contradiction':
          return 2
          
    return dataset['label'].apply(label_mapper).to_list()