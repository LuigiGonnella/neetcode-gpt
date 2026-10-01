import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        # 1. Tokenize by splitting on whitespace: raw_dataset.split()
        # 2. Generate batch_size random start indices using torch.randint()
        #    Range: [0, len(tokens) - context_length)
        # 3. For each index i, X = tokens[i:i+context_length], Y = tokens[i+1:i+1+context_length]
        torch.manual_seed(0)
        

        def tokenize(text: str) -> List[str]: #should be a greedy tokenization on a vocabulary, like we alerady did 

            return text.split()


        n = len(raw_dataset.split())
        tokens = tokenize(raw_dataset)

        random_starts = torch.randint(high = n - context_length, size = (batch_size,)).tolist() #excluded, so until n - C - 1, then when getting until start + context_length + 1 (also excluded) I go until n - 1

        X, Y = [], []

        for start in random_starts:

            X.append(tokens[start:start + context_length])
            Y.append(tokens[start + 1:start + context_length + 1])

        return (X, Y)
