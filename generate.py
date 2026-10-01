import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        # 1. Crop context to context_length if it exceeds it: context[:, -context_length:]
        # 2. Run model(context) -> take last position's logits -> apply softmax(dim=-1)
        # 3. Sample next token with torch.multinomial(probs, 1, generator=generator)
        # 4. Append sampled token to context with torch.cat
        # 5. Map token to character using int_to_char and accumulate result
        # Do not alter the fixed code below — it ensures reproducible test output.

        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        result = []
        for _ in range(new_chars):

            if context.shape[1] > context_length:
                context = context[:, -context_length:] #fake batch pos --> (1, context_length)
            
            full_logits = model(context) #(B, C, vocab_size)
            next_logits = full_logits[:, -1, :] #last word logits (B, 1, vocab_size) over vocab_size
            next_probs = torch.softmax(next_logits, dim =-1) #(B, 1, vocab_size)
            # next_word_id = torch.argmax(next_probs, dim = -1) #greedy sampling
            next_word_id = torch.multinomial(next_probs, 1, generator=generator) #1 sample


            # YOUR CODE (arbitrary number of lines)
            # The line where you call torch.multinomial(). Pass in the generator as well.
            generator.set_state(initial_state)
            # MORE OF YOUR CODE (arbitrary number of lines)
            context = torch.cat([context, next_word_id], dim = -1)
            next_word = int_to_char[next_word_id.item()]

            result.append(next_word)

        return "".join(result)
        # Once your code passes the test, check out the Colab link to see your code generate new Drake lyrics!
