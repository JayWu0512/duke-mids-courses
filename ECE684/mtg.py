import random
from collections import defaultdict, Counter


ALPHA = 0.4
_counts_cache = {}


def finish_sentence(sentence, n, corpus, randomize=False):
    """
    Three Steps:
    1. build models
    2. predict sentence
    3. return result
    """
    # edge cases
    if n < 1:
        raise ValueError("n should larger than 0")
    # if len(sentence) < n:
    #     raise ValueError("Setence length should be larger than n")

    # build n-gram model
    n_grams = build_n_gram_model(corpus, n)

    # start to predict
    result = predict(n_grams, sentence, n, randomize)

    return result


def build_n_gram_model(corpus, n):
    """
    Build n-gram model
    if n=1, model == {():[possible_word1, possible_word2]}
    if n>1, model == {(current_word1, current_word2):[possible_word1, possible_word2]}
    """
    model = defaultdict(list)
    # for unigram model
    model[()].extend(corpus)
    # for bigram/ trigram models
    for k in range(1, n):
        for i in range(len(corpus) - k):
            # use k words to predict the next word
            key = tuple(corpus[i : i + k])
            next_word = corpus[i + k]
            model[key].append(next_word)
    return model


def predict(model, sentence, n, randomize):
    """
    predict next word and return in 10-word sentence
    1. slice n words from the end of the sentence
    2. get all possible words ([possible_word1, possible_word2])
    3. if random, calculate the weight and put into random function
    4. if not random, choose the most possible word (if same proability, get first alphabetically)
    """
    current_key = tuple(sentence[-(n - 1) :])
    while True:
        words = _get_possible_words(model, current_key)
        # edge case, if no possible word, stop predicting
        if not words:
            break
        if randomize == True:
            weights = [_get_score(model, current_key, word) for word in words]
            # edge case, if all weight == 0, random.choices will have error
            if all(weight == 0 for weight in weights):
                next_word = sorted(words, key=str.lower())[0]
            else:
                next_word = random.choices(list(words), weights=weights, k=1)[0]
        else:
            next_word = min(
                words,
                key=lambda word: (-_get_score(model, current_key, word), word.lower()),
            )

        sentence.append(next_word)
        if len(sentence) >= 10:
            break
        if next_word in {".", "?", "!"}:
            break

        # next predict
        if n > 1:
            current_key = tuple(sentence[-(n - 1) :])

    return sentence


def _get_possible_words(model, current_key):
    """
    get all possible words that might be in n grams
    ex. {
    ('I','am','a'): ['student', 'teacher'],
    ('am','a'): ['dog', 'cat'],
    ('a',): ['book', 'pen']
    } ->
    possible_words = {'student', 'teacher', 'dog', 'cat', 'book', 'pen'}
    """
    possible_words = set()
    while True:
        possible_words.update(model.get(current_key, []))
        # break where is no current_key
        if len(current_key) == 0:
            break
        # slice current_key to go to the next round
        current_key = current_key[1:]
    if not possible_words:
        possible_words.update(model.get(current_key, []))
    return possible_words


def _get_score(model, current_key, w, alpha=ALPHA):
    """
    stupid backoff function
    """
    cnts = _counts(model, current_key)
    total = sum(cnts.values())
    if cnts.get(w, 0) > 0 and total > 0:
        return cnts[w] / total
    # for unigram model
    if len(current_key) == 0:
        uni = _counts(model, ())
        total = sum(uni.values())
        return (uni.get(w, 0) / total) if total > 0 else 0.0
    # recursion, slice current key and go to the next
    return alpha * _get_score(model, current_key[1:], w, alpha)


def _counts(model, current_key):
    """
    _counts_cache == Counter({'possible_word1': 2, 'possible_word2': 3})
    """
    if current_key not in _counts_cache:
        _counts_cache[current_key] = Counter(model.get(current_key, []))
    return _counts_cache[current_key]
