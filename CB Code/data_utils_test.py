import data_utils
import pandas as pd
import torch
import unittest

from transformers import RobertaTokenizerFast


class TestDataUtils(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.dataset = pd.DataFrame.from_dict(
            {
                "premise": ["question 0", "question 1", "q3"],
                "hypothesis": ["passage 0", "passage 1", "p3"],
                "idx": [0, 1, 2],
                "label": ['contradiction', 'entailment', 'neutral'],
            }
        )
        self.max_seq_len = 4

    def test_sample(self):
        ## An example of a basic unit test, using class variables initialized in
        ## setUpClass().
        self.assertEqual(self.max_seq_len, 4)

    def test_encode_data(self):
        ## TODO: Write a unit test that asserts that the dimensions and dtype of the
        ## output of encode_data() are correct.
        ## input_ids should have shape [len(self.dataset), self.max_seq_len] and type torch.long.
        ## attention_mask should have the same shape and type.
        input_ids, attention_mask = data_utils.encode_data(self.dataset, self.tokenizer, self.max_seq_len)
        
        dataset_size, seq_length = input_ids.size()
        self.assertEqual(len(self.dataset), dataset_size)
        self.assertEqual(self.max_seq_len, seq_length)
        self.assertEqual(input_ids.dtype, torch.long)

        dataset_size, seq_length = attention_mask.size()
        self.assertEqual(len(self.dataset), dataset_size)
        self.assertEqual(self.max_seq_len, seq_length)
        self.assertEqual(attention_mask.dtype, torch.long)


    def test_extract_labels(self):
        ## TODO: Write a unit test that asserts that extract_labels() outputs the
        ## correct labels, e.g. [1, 0].
        self.assertEqual(data_utils.extract_labels(self.dataset), [2, 1, 0])

if __name__ == "__main__":
    unittest.main()
