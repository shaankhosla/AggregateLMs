import data_utils
import torch

from torch.utils.data import Dataset


class BoolQDataset(Dataset):
    """
    A torch.utils.data.Dataset wrapper for the BoolQ dataset.
    """

    def __init__(self, dataframe, tokenizer, max_seq_length=256):
        """
        Args:
          dataframe: A Pandas dataframe containing the data.
          tokenizer: A transformers.PreTrainedTokenizerFast object that is used to
            tokenize the data.
          max_seq_length: Maximum sequence length to either pad or truncate every
            input example to.
        """
        ## TODO: Use encode_data() from data_utils to store the input IDs and 
        ## attention masks for the data.
        self.encoded_data = data_utils.encode_data(dataframe, tokenizer, max_seq_length)
        ## TODO: Use extract_labels() from data_utils to store the labels.
        self.label_list = data_utils.extract_labels(dataframe)

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, i):
        """
        Returns:
          example: A dictionary containing the input_ids, attention_mask, and
            label for the i-th example, with the values being numeric tensors
            and the keys being 'input_ids', 'attention_mask', and 'labels'.
        """
        ## TODO: Return the i-th example as a dictionary with the keys and values
        ## specified in the function docstring. You should be able to extract the
        ## necessary values from self.encoded_data and self.label_list.
        item = {"input_ids": self.encoded_data[0][i],
               "attention_mask": self.encoded_data[1][i],
               "labels": self.label_list[i]
              }
        return item
