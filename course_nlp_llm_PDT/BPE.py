from collections import defaultdict
#  split words into characters and add </w> at the end
def get_vocab(corpus):
    """Create a vocabulary from the corpus."""
    vocab = {}
    for sentence in corpus:
        for word in sentence.split():
            chars = ' '.join(list(word)) + ' </w>'
            if chars in vocab:
                vocab[chars] += 1
            else:
                vocab[chars] = 1
    return vocab
# vocab format {'word': frequency}
class BytePairEncoding:
    def __init__(self, vocab, num_merges):
        self.vocab = vocab
        self.num_merges = num_merges
        self.bpe_codes = {}


    # get the frequency of all pairs of symbols in the vocabulary 
    def get_pair_counts(self):
        pairs = {}
        for word, freq in self.vocab.items():
            symbols = word.split()
            for i in range (len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                if pair in pairs:
                    pairs[pair] += freq
                else:
                    pairs[pair] = freq
        return pairs
    # pairs = {('a', 'i'): 5, ('i', 's'): 3, ('s', 't'): 2}
    # merge the most frequent pair of symbols in the vocabulary
    def merge_vocab(self, pair):
        bigram = ' '.join(pair) # join the pair by " ", ex: paie = ('a', 'i') -> bigram = 'a i'
        new_vocab = {}
        for word in self.vocab:
            new_word = word.replace(bigram, ''.join(pair)) # find the bigram in the word and replace it with the joined pair, ex: find "a i" -> 'ai
            new_vocab[new_word] = self.vocab[word]
        return new_vocab


    def encode(self):
        for i in range(self.num_merges):
            pairs = self.get_pair_counts()
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            print(f"Step {i+1}: Merging {best}")
            self.vocab = self.merge_vocab(best)
            self.bpe_codes[best] = i

            for word, freq in self.vocab.items():
                print(f"{word}: {freq}")
            print()

        return self.vocab, self.bpe_codes
# Example usage
if __name__ == "__main__":
    corpus = [
        "the cat sits on the mat",
        "the dog lies on the mat",
        "the cat and the dog are friends",
        "dogs and cats are good pets",
        "the pet sits on the rug",
        "a cat sleeps on a chair",
        "a dog barks at the cat",
        "cats and dogs play together"
    ]

    vocab = get_vocab(corpus)
    bpe = BytePairEncoding(vocab, num_merges=10)
    final_vocab, bpe_codes = bpe.encode()

    print("Final Vocabulary:")
    for word, freq in final_vocab.items():
        print(f"{word}: {freq}")

    print("\nBPE Codes:")
    for pair, order in bpe_codes.items():
        print(f"{order}: {pair}")
