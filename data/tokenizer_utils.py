from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        
        tokens = []
        for number in numbers:
            number = list(str(number))
            tokens_number = []

            end = len(number)
            start = 0
            while end > 0 and start < len(number): 
                curr = "".join(number[start:end])
                if curr in vocab:
                    tokens_number.append(curr)
                    start = end
                    end = len(number)
                else:
                    end -= 1
            
            tokens.append(tokens_number)
        
        return tokens

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        tokens_word = []
        
        end = len(text)
        start = 0

        while end > 0 and start < len(text):
            if end <= start:
                raise ValueError(
                    f"No valid token starting at position {start}"
                )
            curr = text[start:end]

            if curr in vocab:
                tokens_word.append(curr)
                start = end
                end = len(text)
            else:
                end -= 1

        return len(tokens_word)

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        n_tokens = self.count_tokens(text, vocab)

        return round(n_tokens / len(text.split()), 4)
