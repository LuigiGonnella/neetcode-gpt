# TOKENIZATION

1) MERGING RULES and VOCABULARY:
We use BPE to train the tokenizer and make it learn the merging rules --> learns what tokens are --> `tokenizer.py`. Then we use these tokens to construct the vocabulary {TOKEN (str): ID (int)} --> `vocab.py` 

2) TOKENIZATION:
We can use greedy left-to-right tokenization where given a TEXT (str) we decompose it in TOKENS (either str or int) --> `tokenizer_utils.py`
