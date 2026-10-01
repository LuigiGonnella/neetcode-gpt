from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        uniques = sorted(list(set(list(text))))
        to_int = range(len(uniques))

        stoi = {char: integer for char, integer in zip(uniques, to_int)}

        itos = {integer: char for char, integer in zip(uniques, to_int)}

        return (stoi, itos)

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping

        integers = [stoi[char] for char in text]

        return integers
        

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        chars = [itos[integer] for integer in ids]

        return "".join(chars)
        


