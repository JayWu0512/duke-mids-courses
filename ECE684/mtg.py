import random
from collections import defaultdict, Counter


ALPHA = 0.4
_counts_cache = {}


def finish_sentence(sentence, n, corpus, randomize=False):

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
    # build n-gram model
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
    current_key = tuple(sentence[-(n - 1) :])
    while True:
        words = _get_possible_words(model, current_key)
        if not words:
            break
        if randomize == True:
            weights = [_get_score(model, current_key, word) for word in words]
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
    possible_words = set()
    while True:
        possible_words.update(model.get(current_key, []))
        if len(current_key) == 0:
            break
        current_key = current_key[1:]
    if not possible_words:
        possible_words.update(model.get(current_key, []))
    return possible_words


def _get_score(model, current_key, w, alpha=ALPHA):
    # stupid backoff
    cnts = _counts(model, current_key)
    total = sum(cnts.values())
    if cnts.get(w, 0) > 0 and total > 0:
        return cnts[w] / total
    if len(current_key) == 0:
        uni = _counts(model, ())
        total = sum(uni.values())
        return (uni.get(w, 0) / total) if total > 0 else 0.0
    return alpha * _get_score(model, current_key[1:], w, alpha)


def _counts(model, current_key):
    if current_key not in _counts_cache:
        _counts_cache[current_key] = Counter(model.get(current_key, []))
    return _counts_cache[current_key]
