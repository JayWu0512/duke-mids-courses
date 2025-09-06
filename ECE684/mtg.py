import random
from collections import defaultdict

ALPHA = 0.4


def finish_sentence(sentence, n, corpus, randomize=False):

    # edge cases
    if n < 1:
        raise ValueError("n should larger than 0")
    if len(sentence) < n:
        raise ValueError("Setence length should be larger than n")

    # build n-gram model
    n_grams = build_n_gram_model(corpus, n)

    # start to predict
    result = predict(n_grams, sentence, n, randomize)

    return result


def build_n_gram_model(corpus, n):
    # build n-gram model
    n_grams = defaultdict(list)
    #  n=3 -> ("I", "love"): ["data", "python", "nlp"]
    for i in range(len(corpus) - n):
        key = tuple(corpus[i : i + n - 1])
        next_word = corpus[i + n - 1]
        n_grams[key].append(next_word)
    return n_grams


def predict(n_grams, sentence, n, randomize):
    current_key = tuple(sentence[-(n - 1) :])
    while True:
        if current_key not in n_grams:
            break
        next_words = n_grams[current_key]

        if randomize == True:
            next_word = random.choices(next_words)
        else:
            next_word = next_words[0]
        sentence.append(next_word)

        if len(sentence) >= 10:
            break
        if next_word in {".", "?", "!"}:
            break

        # next predict
        if n > 1:
            current_key = tuple(sentence[-(n - 1) :])

    return sentence
