import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        uniques = sorted(list(set([word for el in positive + negative for word in el.split() ])))
        ids = {word: float(idx) for (idx, word) in enumerate(uniques, start = 1)}

        out = []

        for sentence in positive + negative:
            encoded = [ids[word] for word in sentence.split()]
            out.append(torch.tensor(encoded))

        return nn.utils.rnn.pad_sequence(out, batch_first=True)
