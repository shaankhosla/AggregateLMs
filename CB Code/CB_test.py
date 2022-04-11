import pandas as pd
import torch
import unittest

from CB import CBDataset
from transformers import RobertaTokenizerFast


class TestCBDataset(unittest.TestCase):
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
        self.CB_dataset = CBDataset(
            self.dataset, self.tokenizer, self.max_seq_len
        )

    def test_len(self):
        ## TODO: Test that the length of self.boolq_dataset is correct.
        ## len(self.boolq_dataset) should equal len(self.dataset).
        self.assertEqual(len(self.CB_dataset), len(self.dataset))


    def test_item(self):
        ## TODO: Test that, for each element of self.boolq_dataset, 
        ## the output of __getitem__ (accessible via self.boolq_dataset[idx])
        ## has the correct keys, value dimensions, and value types.
        ## Each item should have keys ["input_ids", "attention_mask", "labels"].
        ## The input_ids and attention_mask values should both have length self.max_seq_len
        ## and type torch.long. The labels value should be a single numeric value.
        keyset = {"input_ids", "attention_mask", "labels"}

        for element in self.CB_dataset:

            self.assertEqual(set(element.keys()), keyset)
            
            self.assertEqual(len(element['input_ids']), self.max_seq_len)
            self.assertEqual(len(element['attention_mask']), self.max_seq_len)
            
            self.assertEqual(element['input_ids'].dtype, torch.long)
            self.assertEqual(element['attention_mask'].dtype, torch.long)
            
            self.assertEqual(type(element['labels']), int)

if __name__ == "__main__":
    unittest.main()
