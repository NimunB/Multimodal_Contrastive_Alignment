# Multimodal Contrastive Alignment: CIFAR-100 Semantic Expansion with Skip-Gram Embedding

**Nimun Kaur Bajwa**

---

## Overview

This project produces a single embedding matrix containing the union of the original Visual Genome vocabulary (~455 words) and all 100 CIFAR-100 class labels, yielding a final vocabulary of **523 words**. The approach uses an **Evolutionary Insertion** strategy: a robust Skip-Gram base model is trained first, and missing CIFAR-100 words are then inserted using a Genetic/Evolutionary fitness function.

The best model checkpoint is saved as `best_skipgram_523words.pth` and is loaded by the entry-point function `build_my_embeddings` in `cw2.py`.

---

## Repository Structure

```
.
├── cw2.py                        # Entry point — contains build_my_embeddings()
├── best_skipgram_523words.pth    # Best model checkpoint (required for loading)
├── vg_text/                      # Visual Genome corpus
│   └── vg_text_cifar.txt         # Augmented corpus with CIFAR-100 word substitutions
├── report.pdf                    # Full scientific report
└── README.md
```

---

## Quickstart

```python
from cw2 import build_my_embeddings

vocab, embeddings = build_my_embeddings()
# vocab: list of 523 words
# embeddings: np.ndarray of shape (523, 256)
```

The function loads `best_skipgram_523words.pth` relative to the current working directory. **Do not move or rename this file.**

---

## Methodology

Development was structured in two sequential phases, each consisting of key logic improvements followed by hyperparameter grid searches.

### Phase 1 — Optimising the Base Skip-Gram Embedding

**Key Logic Improvements**

- **Excluding all contexts from negatives:** Prevents contradictory training signals where a word's own context words are sampled as negatives.
- **Stopword isolation:** Stopwords are only included as context when the target word is itself a stopword. This prevents commonly used words from collapsing into content-word space and encourages a distinct stopword cluster.

**Grid Search 1 — Architecture Parameters**

| Parameter           | Values Tested   |
|---------------------|-----------------|
| Embedding Dimension | 128, 256        |
| Number of Negatives | 8, 13, 18, 23   |
| Context Size        | 1, 2, 3         |

Fixed across search: batch size 1024, 20 epochs, learning rate 0.001, dropout 0.2, weight decay 1e-4, label smoothing 0.1, patience 5.

Results ranked by two metrics (see Evaluation section below):
- **Best by Signal-to-Noise:** Dimension 256, Negatives 8, Context 1 (score: 0.0243)
- **Best by CIFAR-Neighbours:** Dimension 256, Negatives 18, Context 1 (score: 0.3077)

CIFAR-Neighbours was prioritised as it more directly measures semantic clustering quality. **Carried forward: Dimension 256, Negatives 18, Context 1.**

**Grid Search 2 — Regularisation Parameters**

| Parameter       | Values Tested      |
|-----------------|--------------------|
| Learning Rate   | 0.01, 0.03, 0.05   |
| Dropout         | 0.1, 0.2, 0.3      |
| Weight Decay    | 1e-5, 1e-4, 1e-3   |
| Label Smoothing | 0.0, 0.1, 0.2      |

Two configurations appeared in the top 6 for both metrics simultaneously:
1. Learning Rate 0.03, Dropout 0.2, Weight Decay 1e-4, Label Smoothing 0.0
2. Learning Rate 0.03, Dropout 0.1, Weight Decay 1e-5, Label Smoothing 0.1

Configuration 1 was selected on the basis of higher mean rank across both metrics.

**Final Phase 1 settings:** Dimension 256, Negatives 18, Context 1, LR 0.03, Dropout 0.2, Weight Decay 1e-4, Label Smoothing 0.0.

---

### Phase 2 — Inserting CIFAR-100 Words via Evolutionary Strategy

**Key Logic Improvements**

- **Anchor creation:** Each new CIFAR-100 word is assigned anchor words already present in the embedding space. Anchors serve as fixed reference points in the fitness function, pulling the new word's vector towards its correct semantic neighbourhood.
- **Corpus augmentation:** CIFAR-100 words that do not appear in the raw `vg_text` corpus (due to formatting differences or rarity) are handled by creating an augmented copy of the corpus and substituting near-equivalent terms (e.g. `aquarium` → `aquarium_fish`, `rat` → `shrew`). This allows meaningful co-occurrence contexts to be constructed for otherwise unseen words.
- **Stopword scoping and sentence boundaries:** Stopwords are only used as context for stopword targets, and context windows are restricted to sentence boundaries to avoid cross-sentence noise.

**Grid Search 1 — Anchor Count**

| Anchors | Signal-to-Noise | CIFAR-Neighbours |
|---------|-----------------|------------------|
| Limited (1–2) | 0.2743 | 1.5698 |
| Many (4–5)    | 0.1703 | 1.4767 |

Using a limited set of anchors outperformed many anchors on both metrics. **Carried forward: limited anchors.**

**Grid Search 2 — Fitness Function Weights**

The fitness function combines three components: corpus co-occurrence similarity, magnitude matching, and anchor similarity. Six weight combinations were tested:

| Corpus | Magnitude | Anchor | Signal-to-Noise | CIFAR-Neighbours |
|--------|-----------|--------|-----------------|------------------|
| 0.2    | 0.2       | 0.6    | **0.5019**      | **1.6163**       |
| 0.3    | 0.3       | 0.4    | 0.4286          | 1.6047           |
| 0.4    | 0.25      | 0.35   | 0.3264          | 1.3837           |
| 0.5    | 0.15      | 0.35   | 0.2997          | 1.5465           |
| 0.6    | 0.15      | 0.25   | 0.2728          | 1.5698           |
| 0.5    | 0.25      | 0.25   | 0.2683          | 1.5116           |

The configuration `[Corpus: 0.2, Norm: 0.2, Anchor: 0.6]` achieved the highest rank on both metrics and was selected as the **final embedding**.

---

## Evaluation Metrics

Two custom metrics were used throughout development to compare embeddings:

**Signal-to-Noise Ratio**
Mean intra-CIFAR superclass cosine similarity minus the mean cosine similarity between 100 randomly sampled non-stopwords. Maximising this encourages tight intra-class clustering while preventing general embedding collapse. The baseline is restricted to non-stopwords to avoid solutions where stopwords are simply pushed far away while all content words collapse together.

**CIFAR-Neighbours**
The average number of nearest neighbours (by cosine similarity) of each CIFAR-100 word that belong to the same CIFAR-100 superclass. Higher values indicate that semantically related concepts are correctly situated near one another in the embedding space.

---

## Final Model Configuration

| Setting             | Value  |
|---------------------|--------|
| Embedding Dimension | 256    |
| Number of Negatives | 18     |
| Context Size        | 1      |
| Learning Rate       | 0.03   |
| Dropout             | 0.2    |
| Weight Decay        | 1e-4   |
| Label Smoothing     | 0.0    |
| Corpus Weight       | 0.2    |
| Norm Weight         | 0.2    |
| Anchor Weight       | 0.6    |
| Anchors per word    | 1–2    |
| Vocabulary size     | 523    |

---

## Notes

- Training code is included for reference but **will not be re-run during assessment**. Only `best_skipgram_523words.pth` is loaded.
- The checkpoint must remain in the root directory alongside `cw2.py`.
