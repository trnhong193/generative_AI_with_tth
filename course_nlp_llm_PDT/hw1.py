def unigram_tokenizer(text):
    tokens = text.lower().split()
    return tokens

def bigram_tokenizer(text):
    tokens = text.lower().split()   
    bigrams_tokens = []
    for i in range(len(tokens)-1):
        bigrams = (tokens[i], tokens[i+1])
        bigrams_tokens.append(bigrams)
    return bigrams_tokens

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
for sentence in corpus:
    print("Sentence:", sentence)
    print("Unigram tokens:", unigram_tokenizer(sentence))
    print("Bigram tokens:", bigram_tokenizer(sentence))