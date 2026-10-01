from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed

        merges = [] #result array


        #1
        corpus_tokens = list(corpus) #split corpus in single tokens
        

        #2
        for step in range(num_merges):
            if len(corpus_tokens) < 2:
                break
                
            pairs = list(zip(corpus_tokens[:-1], corpus_tokens[1:])) #get pairs of adjacent tokens (a, b)
            
            counter = Counter(pairs) #get dict with token pair: frequency
            inv = defaultdict(list) #we want frequency: list(token pair)
            sorted_keys = sorted(counter, key = lambda x: counter[x], reverse=True)
            max_count = counter[sorted_keys[0]] #tracks max_frequency, take frequency of most frequent ([0] element)

            for max_key in sorted_keys: #iterate over sorted dict of pairs from most frequent
                if counter[max_key] < max_count: #we just want max_keys
                    break

                inv[counter[max_key]].append(max_key) #add in freq: list(key)

            inv[max_count].sort() #sort max keys (pairs)
            print(counter)

            most_freq_key = inv[max_count][0] #max freq pair
            merges.append(list(most_freq_key))

            print(most_freq_key)
            tmp = [] 
            skip = False #do not add twice the same pair
            for i, el in enumerate(zip(corpus_tokens[:-1], corpus_tokens[1:])):
                a, b = el #take the pair

                if skip:
                    skip = False

                    if i == len(corpus_tokens) - 2: #if last pair -- add also second token in the pair
                        tmp.append(b)
                    continue
                if (a, b) != most_freq_key: #if pair is not most frequent
                    tmp.append(a) #do not merge, add first (the second will be added in the next iteration)

                    if i == len(corpus_tokens) - 2: #if last pair -- add also second token in the pair
                        tmp.append(b)
                else: #max pair
                    tmp.append(a + b) #merge
                    skip = True #do not analyze (b, c) after (a, b) since b is not a single token anymore -- we merged it into 'ab'

            corpus_tokens = tmp
            print(corpus_tokens)


        return merges



                   


