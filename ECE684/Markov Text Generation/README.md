# Markov_Text_Generation


## 1) Function Overview

`finish_sentence(sentence, n, corpus, randomize=False)` extends a seed sentence (list/tuple of tokens) **until** it meets the first end‑of‑sentence token (`. ? !`) **or** reaches **10** total tokens.

- If `randomize=False` (deterministic): at each step choose the **highest‑score** next token; if there’s a tie, choose the token that is **alphabetically first** (case‑insensitive).
- If `randomize=True` (stochastic): draw the next token **randomly** using **Stupid Backoff** scores as sampling weights.

No smoothing is used. Backoff uses discount **α = 0.4**.

---

## 2) Method (How it works)

### 2.1 n‑gram model (histories 0 … n−1)
Build a dictionary `model: history -> list of next tokens` for **all** history lengths from 0 to `n−1`:

- `()` (empty history) holds the full corpus tokens (unigram base).
- `(w1,) -> [w2, …]`, `(w1, w2) -> [w3, …]`, …, up to length `n−1` histories.

This allows backing off from longer histories to shorter ones when the longer history wasn’t observed.

### 2.2 Stupid Backoff scoring
For history `h` and candidate next token `w`:
```
SB(w | h) = 
  count(h, w) / sum_u count(h, u)            
  if count(h, w) > 0
  α * SB(w | h[1:])                           otherwise
```
When `h` becomes the empty history `()`, we use the unigram ML probability `count(w) / total`.
We set **α = 0.4**.

### 2.3 Selection rule
- **Deterministic:** choose `argmax_w SB(w|h)`; break ties by alphabet (`min` on `w.casefold()`).
- **Stochastic:** sample with weights proportional to `SB(w|h)`.

---

## 3) Function Signature & Stopping

```python
finish_sentence(
    sentence: list[str] | tuple[str, ...],
    n: int,
    corpus: list[str] | tuple[str, ...],
    randomize: bool = False
) -> list[str]
```
Requirements:
- `n >= 1` and the seed contains at least `n-1` tokens.
- Stop once `. ? !` is generated **or** length reaches 10.

---

## 4) Example Applications (Deterministic & Stochastic)

Below we show **both deterministic and stochastic modes**, with **multiple seeds** and **different `n`**, on a **toy corpus** and on **Austen’s _Sense and Sensibility_**.

> For stochastic examples we set `random.seed(...)` to make runs reproducible.

### 4.1 Setup

```python
import random, nltk
from mtg import finish_sentence

# Toy corpus (pre-tokenized)
toy = "the cat sat on the mat . the cat ate the fish . the dog sat on the rug . the dog ate the bone .".split()

# Austen corpus (lowercased + tokenized)
nltk.download('gutenberg'); nltk.download('punkt')
austen = nltk.word_tokenize(nltk.corpus.gutenberg.raw('austen-sense.txt').lower())
```

### 4.2 Deterministic mode (`randomize=False`)

#### (A) Required test case (trigram, `n=3`)
```python
seed = ['she', 'was', 'not']
finish_sentence(seed.copy(), 3, austen, randomize=False)
```
**Expected output:**
```
['she', 'was', 'not', 'in', 'the', 'world', ',', 'and', 'the', 'two']
```

#### (B) Toy corpus, `n=3`, different seeds
```python
finish_sentence(['the','cat'], 3, toy, randomize=False)
```
Output:
```
['the','cat','ate','the','bone','.']
```
```python
finish_sentence(['the','dog'], 3, toy, randomize=False)
```
Output:
```
['the','dog','ate','the','bone','.']
```

#### (C) Toy corpus, **different n** (`n=2`, bigram)
```python
finish_sentence(['the','cat'], 2, toy, randomize=False)
```
Output (stops at 10 tokens because no punctuation is reached earlier):
```
['the','cat','ate','the','cat','ate','the','cat','ate','the']
```

### 4.3 Stochastic mode (`randomize=True`)

> Weights = Stupid Backoff scores; set a seed for reproducibility.

```python
random.seed(7)
finish_sentence(['the','cat'], 3, toy, randomize=True)
```
One possible output:
```
['the','cat','sat','on','the','mat','.']
```

```python
random.seed(9)
finish_sentence(['the','dog'], 3, toy, randomize=True)
```
One possible output:
```
['the','dog','sat','on','the','rug','.']
```

```python
random.seed(123)
finish_sentence(['she','was','not'], 3, austen, randomize=True)
```
One example output (your exact line may differ):
```
['she','was','not', ...]
```

---

## 5) Observations

- **Effect of `n`:** larger `n` gives more context‑specific predictions but triggers backoff more when data are sparse.  
- **Deterministic vs Stochastic:** deterministic is repeatable (good for unit tests); stochastic explores diverse outputs (good for generation).  
- **Backoff α=0.4:** ensures coverage (never stuck) while preferring longer‑history evidence; ties resolved alphabetically.

---

## 6) How to Run

1. Place implementation in `mtg.py`.
2. (First time) download NLTK data:
   ```python
   import nltk
   nltk.download('gutenberg'); nltk.download('punkt')
   ```
3. Run the examples above in a Python shell or notebook.

---

## 7) Minimal test script (`test_mtg.py`)

```python
import nltk, random
from mtg import finish_sentence

def test_austen_trigram_deterministic():
    nltk.download('gutenberg'); nltk.download('punkt')
    corpus = nltk.word_tokenize(nltk.corpus.gutenberg.raw('austen-sense.txt').lower())
    seed = ['she','was','not']
    out = finish_sentence(seed.copy(), 3, corpus, randomize=False)
    assert out == ['she','was','not','in','the','world',',','and','the','two']

def test_stops_by_length_or_punct():
    toy = "the cat sat on the mat . the cat ate the fish . the dog sat on the rug . the dog ate the bone .".split()
    out = finish_sentence(['the','cat'], 2, toy, randomize=False)
    assert len(out) == 10 or out[-1] in {'.','?','!'}

if __name__ == '__main__':
    test_austen_trigram_deterministic()
    test_stops_by_length_or_punct()
    print('OK')
```

---

## 8) Implementation Notes (highlights from `mtg.py`)

- Build histories for lengths **0 … n−1** (include `()` as unigram base).
- Stupid Backoff scoring with **α = 0.4**; no smoothing.
- Deterministic selection (tie‑break alphabetically):
```python
best = min(cands, key=lambda w: (-score[w], w.casefold()))
```
- Stochastic selection:
```python
next_word = random.choices(cand_list, weights=weights, k=1)[0]
```
- Cache counts per history to avoid recomputation.