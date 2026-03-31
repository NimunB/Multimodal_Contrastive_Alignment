"""
Simple Genetic Algorithm for Word Embedding Insertion
======================================================================
A (1+λ) Evolution Strategy for inserting new word embeddings into a trained
Skip-Gram model while preserving the existing embedding space structure.
"""

import torch
import numpy as np
import re
from collections import Counter
from typing import Dict, List, Tuple, Optional, Set

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torchvision

from lab6 import SkipGramModel, find_similar_words
from lab2 import process_text_network

import unittest
import tempfile
import os


# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================

def load_trained_model(model_path: str, vocab_size: int, 
                       embedding_dim: int, dropout: float) -> Tuple[torch.nn.Module, np.ndarray]:
    """Load trained Skip-Gram model and extract embeddings."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    model = SkipGramModel(vocab_size=vocab_size, embedding_dim=embedding_dim, dropout=dropout).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        embeddings_tensor = model.get_embeddings()
        embeddings = (embeddings_tensor.cpu().numpy() if isinstance(embeddings_tensor, torch.Tensor) 
                     else embeddings_tensor).astype(np.float32)
    
    print(f"✓ Loaded model: {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")
    return model, embeddings


def create_mappings(nodes: List[str]) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, np.ndarray]]:
    """Create word-to-index and index-to-word mappings."""
    word_to_idx = {word: idx for idx, word in enumerate(nodes)}
    idx_to_word = {idx: word for idx, word in enumerate(nodes)}
    return word_to_idx, idx_to_word


def compute_embedding_stats(embeddings: np.ndarray) -> Dict[str, float]:
    """Compute statistics needed for fitness evaluation."""
    norms = np.linalg.norm(embeddings, axis=1)
    return {
        'mean_norm': np.mean(norms),
        'std_norm': np.std(norms),
        'global_std': np.std(embeddings)
    }


def get_cifar100_vocabulary() -> List[str]:
    """Download CIFAR-100 and extract class names."""
    print("\nLoading CIFAR-100 vocabulary...")
    dataset = torchvision.datasets.CIFAR100(root='./cifar100_data', train=True, download=True)
    print(f"✓ CIFAR-100 vocabulary loaded: {len(dataset.classes)} classes")
    return dataset.classes


def analyze_vocabulary_overlap(cifar_vocab: List[str], network_vocab: List[str]) -> List[str]:
    """Analyze overlap between CIFAR-100 and network vocabulary."""
    cifar_set, network_set = set(cifar_vocab), set(network_vocab)
    overlapping = sorted(list(cifar_set.intersection(network_set)))
    missing = sorted(list(cifar_set - network_set))
    
    print(f"\n{'='*70}")
    print("VOCABULARY OVERLAP ANALYSIS")
    print(f"{'='*70}")
    print(f"CIFAR-100 vocabulary: {len(cifar_set)} classes")
    print(f"Network vocabulary: {len(network_set)} words")
    print(f"Overlapping words: {len(overlapping)} ({len(overlapping)/len(cifar_set)*100:.1f}%)")
    print(f"Missing from network: {len(missing)}")
    if overlapping:
        print(f"\nFound: {', '.join(overlapping)}")
    if missing:
        print(f"\nMissing: {', '.join(missing)}")
    print(f"{'='*70}\n")
    
    return missing


# ============================================================================
# CONTEXT EXTRACTION
# ============================================================================

# Implemented in Lab 7 (Nov 20) - AM
def extract_word_contexts(
    text_file: str,
    target_words: List[str],
    vocab_set: Set[str],
    window: int = 5
) -> Dict[str, Counter]:
    """
    Extract co-occurrence context statistics for target words from a text corpus.
    
    This function reads a corpus file line-by-line and tracks which words appear
    near specified target words. For each target word, it counts how many times
    each vocabulary word appears within a window around it.
    
    Args:
        text_file: Path to the corpus text file to analyze.
        target_words: List of words to extract contexts for.
        vocab_set: Set of valid vocabulary words (only count these as contexts).
        window: Number of words to look on each side of the target word.
    
    Returns:
        A dictionary mapping each target word to a Counter of context words and
        their frequencies.
        
    Example:
        >>> extract_word_contexts('corpus.txt', ['king', 'queen'], vocab, window=2)
        {'king': Counter({'royal': 5, 'crown': 3}), 
         'queen': Counter({'royal': 4, 'throne': 2})}
    
    Implementation guidelines:
    --------------------------
    1. Initialize a dictionary `{word: Counter()}` for each target word.
    2. Convert `target_words` to a set for fast lookup.
    3. Stream through the file line-by-line (efficient for large corpora).
    4. For each line:
        - Tokenize using lowercase alphabetic words (regex: r"\\b[a-z]+\\b").
        - For each token that matches a target word:
            * Extract up to `window` tokens on both sides.
            * Exclude the target word itself.
            * Retain only context words that appear in `vocab_set`.
            * Update the Counter for that target word.
    5. Handle edge cases: empty lines, start/end of token lists.
    6. Optionally print progress (e.g., every 50,000 lines) for user feedback.
    7. Return the dictionary of Counters.
    """
    
    # Initialize contexts dictionary with a Counter for each target word.
    contexts = {word: Counter() for word in target_words}
    target_set = set(target_words)
    
    # Implement the corpus scanning and context extraction logic.
    with open(text_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 50000 == 0 and i>0:
                print(f' .....processed {i:,} lines')

            tokens = re.findall(r'\b[a-z]+\b', line.lower()) # lowercases wat's not in engl vocab
            for idx, token in enumerate(tokens):
                if token in target_set: # look for its context
                    start, end = max(0, idx - window), min(len(tokens), idx + window + 1)
                    valid_contexts = [t for t in tokens[start:idx] + tokens[idx+1:end] if t in vocab_set] # check if in model
                    contexts[token].update(valid_contexts)
    print(f" Complete \n\n Context statistics")
    for word in target_words:
        print(f"  {word:10s}: {sum(contexts[word].values()):6d} contexts, {len(contexts[word]):3d} unique words") # prints how big context is, and how many unique words
    
    return contexts






# ============================================================================
# FITNESS FUNCTION
# ============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


# Implemented in Lab 7 (Nov 20) - PM
def compute_fitness(
    vec: np.ndarray,
    word: str,
    ctx_vecs: Optional[np.ndarray],
    ctx_weights: Optional[np.ndarray],
    neg_vecs: np.ndarray,
    anchor_vecs: Optional[np.ndarray],
    stats_dict: Dict[str, float],
    weights: Dict[str, float]
) -> float:
    """
    Compute a three-term fitness score for a candidate word embedding vector.
    
    This function evaluates how well a candidate vector fits the learned 
    embedding space by combining three complementary metrics:
    1. Corpus likelihood (how well it predicts observed contexts)
    2. Norm matching (how similar its magnitude is to typical embeddings)
    3. Anchor similarity (how similar it is to known reference words)
    
    Args:
        vec: Candidate embedding vector to evaluate.
        word: Target word (for reference, not used in computation).
        ctx_vecs: Context word vectors that co-occur with the target word.
                  Shape: (n_contexts, embedding_dim). May be None if no contexts.
        ctx_weights: Weights for each context (e.g., co-occurrence counts).
                     Shape: (n_contexts,). May be None if no contexts.
        neg_vecs: Negative sample vectors (words that don't co-occur).
                  Shape: (n_negatives, embedding_dim).
        anchor_vecs: Pre-normalized vectors of anchor words for comparison.
                     Shape: (n_anchors, embedding_dim). May be None.
        stats_dict: Dictionary containing embedding statistics:
                    - 'mean_norm': Average L2 norm of embeddings in the space
                    - 'std_norm': Standard deviation of embedding norms
                    - 'global_std': Global standard deviation (if needed)
        weights: Dictionary of weights for each fitness component:
                 - 'corpus': Weight for corpus likelihood term
                 - 'norm': Weight for norm matching term
                 - 'anchor': Weight for anchor similarity term
    
    Returns:
        Combined fitness score in the range [0, 1], where higher is better.
        
    Example:
        >>> vec = np.array([0.5, -0.3, 0.8, 0.1])
        >>> stats = {'mean_norm': 1.0, 'std_norm': 0.2, 'global_std': 0.5}
        >>> weights = {'corpus': 0.5, 'norm': 0.3, 'anchor': 0.2}
        >>> fitness = compute_fitness(vec, 'king', ctx_vecs, ctx_weights, 
        ...                           neg_vecs, anchor_vecs, stats, weights)
        >>> print(f"Fitness: {fitness:.4f}")
        Fitness: 0.7234
    
    Implementation guidelines:
    --------------------------
    Term 1 - Corpus Likelihood (L_corpus_norm):
        - For positive contexts: sum over ctx_weights * log(sigmoid(ctx_vecs · vec))
        - For negative samples: sum over log(sigmoid(-neg_vecs · vec))
        - Add small epsilon (1e-10) inside log for numerical stability
        - Normalize by total samples, then apply sigmoid to map to [0, 1]
        - Default to 0.5 if no samples available
        
    Term 2 - Norm Match (S_norm):
        - Compute L2 norm of the candidate vector
        - Use Gaussian similarity: exp(-((norm - mean_norm)² / (2 * std_norm²)))
        - This rewards vectors with norms close to the typical embedding norm
        
    Term 3 - Anchor Similarity (S_anchor):
        - Normalize the candidate vector (divide by its norm + epsilon)
        - Compute dot products with all anchor vectors (they're pre-normalized)
        - Take the mean similarity across all anchors
        - Default to 0.5 if no anchors provided
        
    Final score:
        - Weighted sum: weights['corpus'] * L_corpus_norm + 
                       weights['norm'] * S_norm + 
                       weights['anchor'] * S_anchor
    
    Notes:
        - Handle None values for optional parameters (ctx_vecs, ctx_weights, anchor_vecs)
        - Use vectorized NumPy operations for efficiency
        - Add small epsilon values to prevent division by zero
    """
    vec_norm = np.linalg.norm(vec)

    # Corpus likelihood term
    L_corpus = 0.0
    if ctx_vecs is not None:
        L_corpus += np.sum(ctx_weights * np.log(sigmoid(np.dot(ctx_vecs, vec)) + 1e-10)) # Adding epsilon bc log(0) is undefined
    L_corpus += np.sum(np.log(sigmoid(-np.dot(neg_vecs, vec)) + 1e-10)) # Negative for a reason (don't want them)
    # q: why don't we have a weight term here
    # hint: add a weight for negative
    total_samples = (np.sum(ctx_weights) if ctx_vecs is not None else 0) + len(neg_vecs)
    L_corpus_norm = sigmoid(L_corpus / total_samples) if total_samples > 0 else 0.5 # does better w 0.5 as opposed to 0 - play around with it

    # embedding belonging term
    S_norm = np.exp(-((vec_norm - stats_dict['mean_norm']) ** 2 / (2 * stats_dict['std_norm']))) # q: Gaussian correct? Sure denom isn't this: (2 * std_norm**2))?

    # anchor term (words we are saying what things should be close to)
    S_anchor = 0.5 # what he was sayingg
    if anchor_vecs is not None:
        S_anchor = np.mean(np.dot(anchor_vecs, vec / (vec_norm + 1e-10)))

    return (weights['corpus'] * L_corpus_norm + weights['norm'] * S_norm + weights['anchor'] * S_anchor)


# ============================================================================
# GENETIC ALGORITHM (1+λ) EVOLUTION STRATEGY
# ============================================================================

# Implemented in Lab 7 (Nov 20) - AM
def initialize_embedding(
    word: str,
    contexts: Dict[str, Counter],
    embeddings: np.ndarray,
    word_to_idx: Dict[str, int]
) -> np.ndarray:
    """
    Initialize an embedding vector for a word using corpus bootstrap.
    
    This function creates an initial embedding by computing a weighted average
    of the embeddings of words that frequently co-occur with the target word.
    This provides a data-driven starting point that places the new word near
    semantically related words in the embedding space.
    
    Args:
        word: Target word to initialize an embedding for.
        contexts: Dictionary mapping words to their co-occurrence contexts.
                  Each value is a Counter with {context_word: count}.
        embeddings: Pre-trained embedding matrix. Shape: (vocab_size, embedding_dim).
        word_to_idx: Dictionary mapping words to their row indices in embeddings.
    
    Returns:
        Initial embedding vector for the word. Shape: (embedding_dim,).
        
    Example:
        >>> contexts = {'king': Counter({'queen': 50, 'royal': 30, 'castle': 20})}
        >>> embeddings = np.random.randn(1000, 300)  # 1000 words, 300 dims
        >>> word_to_idx = {'queen': 0, 'royal': 1, 'castle': 2, ...}
        >>> vec = initialize_embedding('king', contexts, embeddings, word_to_idx)
        >>> vec.shape
        (300,)
    
    Implementation guidelines:
    --------------------------
    1. Handle the no-context case:
       - If the word has no contexts (empty Counter), return the mean of all
         embeddings as a neutral starting point
    
    2. Get top context words:
       - Extract the top 20 most frequent context words using Counter.most_common()
       - This focuses on the strongest statistical relationships
    
    3. Compute weighted average:
       - Calculate the total weight (sum of all counts)
       - For each context word that exists in word_to_idx:
           * Get its embedding vector
           * Weight it by (count / weight_sum)
           * Add to running sum
    
    4. Validate the result:
       - Check if the resulting vector has non-zero norm
       - If zero (e.g., no valid context words found), fall back to mean embedding
    
    Notes:
        - Some context words may not be in word_to_idx; skip these
        - The weighted average naturally places the new word near its contexts
        - Using top 20 contexts balances informativeness with noise reduction
    """
    if word not in contexts or not contexts[word]: # Q: Added this: word not in contexts
        return np.mean(embeddings, axis=0)
        
    top_contexts = contexts[word].most_common(20)
    weight_sum = sum(count for _, count in top_contexts)
    vec_sum = sum((count/weight_sum) * embeddings[word_to_idx[w]] for w, count in top_contexts if w in word_to_idx)
  
    return vec_sum if np.linalg.norm(vec_sum) > 0 else np.mean(embeddings, axis=0)


# Implemented in Lab 7 (Nov 20) - PM
def precompute_fitness_vectors(
    word: str,
    contexts: Dict[str, Counter],
    embeddings: np.ndarray,
    word_to_idx: Dict[str, int],
    vocab_list: List[str],
    anchors: Dict[str, List[str]],
    num_negatives: int = 15
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """
    Precompute all vectors needed for fitness evaluation.
    
    This function extracts and prepares the three types of vectors used in
    fitness computation: positive context vectors, negative sample vectors,
    and anchor vectors. Precomputing these vectors once improves efficiency
    when evaluating fitness multiple times during optimization.
    
    Args:
        word: Target word being optimized.
        contexts: Dictionary mapping words to their co-occurrence contexts.
                  Each value is a Counter with {context_word: count}.
        embeddings: Pre-trained embedding matrix. Shape: (vocab_size, embedding_dim).
        word_to_idx: Dictionary mapping words to their row indices in embeddings.
        vocab_list: List of all vocabulary words (for negative sampling).
        anchors: Dictionary mapping words to lists of semantically related anchor words.
        num_negatives: Number of negative samples to draw (default: 15).
    
    Returns:
        Tuple of (ctx_vecs, ctx_weights, neg_vecs, anchor_vecs):
        - ctx_vecs: Context word embeddings. Shape: (n_contexts, dim) or None.
        - ctx_weights: Normalized context weights. Shape: (n_contexts,) or None.
        - neg_vecs: Negative sample embeddings. Shape: (num_negatives, dim).
        - anchor_vecs: Normalized anchor embeddings. Shape: (n_anchors, dim) or None.
        
    Example:
        >>> contexts = {'king': Counter({'queen': 50, 'royal': 30})}
        >>> anchors = {'king': ['queen', 'monarch', 'ruler']}
        >>> ctx_v, ctx_w, neg_v, anc_v = precompute_fitness_vectors(
        ...     'king', contexts, embeddings, word_to_idx, vocab_list, anchors
        ... )
        >>> ctx_v.shape  # Positive contexts
        (2, 300)
        >>> neg_v.shape  # Negative samples
        (15, 300)
    
    Implementation guidelines:
    --------------------------
    Part 1 - Positive Context Vectors:
        - Initialize ctx_vecs and ctx_weights to None (for no-context case)
        - If the word has contexts:
            * Iterate through contexts[word].items()
            * For each context word that exists in word_to_idx:
              - Collect its embedding vector
              - Collect its count
            * If any valid contexts found:
              - Convert lists to numpy arrays
              - Normalize weights to sum to 1.0
    
    Part 2 - Negative Sample Vectors:
        - Randomly sample num_negatives words from vocab_list (without replacement)
        - Look up their embeddings and stack into an array
        - Shape should be (num_negatives, embedding_dim)
    
    Part 3 - Anchor Vectors:
        - Initialize anchor_vecs to None (for no-anchor case)
        - If the word has anchors defined:
            * Filter to only anchors that exist in word_to_idx
            * If any valid anchors found:
              - Collect their embeddings into an array
              - Normalize each vector to unit length (L2 norm = 1)
              - Use np.linalg.norm with axis=1, keepdims=True
              - Add small epsilon (1e-10) to prevent division by zero
    
    Notes:
        - Handle missing words gracefully (skip if not in word_to_idx)
        - Return None for optional components if no valid data available
        - Negative samples should be random to avoid bias
        - Anchor normalization enables direct cosine similarity via dot product
    """
    # Positive context
    ctx_vecs, ctx_weights = None, None
    if word in contexts and contexts[word]: # Better error checking (word in contexts -> avoids key error)
        vecs, weights = [], []
        for w, count in contexts[word].items():
            if w in word_to_idx:
                vecs.append(embeddings[word_to_idx[w]])
                weights.append(count)
        if vecs: # Q: Moved out of for loop
            ctx_vecs = np.array(vecs)
            ctx_weights = np.array(weights) / sum(weights)
    # negative samples
    neg_words = np.random.choice(vocab_list, num_negatives, replace=False) # random choice, messes our global minima
    neg_vecs = np.array([embeddings[word_to_idx[n]] for n in neg_words])

    # anchors
    anchor_vecs = None
    if word in anchors:
        anchor_words = [a for a in anchors[word] if a in word_to_idx]
        if anchor_words:
            anchor_vecs = np.array([embeddings[word_to_idx[a]] for a in anchor_words])
            anchor_vecs = anchor_vecs / (np.linalg.norm(anchor_vecs, axis=1, keepdims=True) + 1e-10)
    
    return (ctx_vecs, ctx_weights, neg_vecs, anchor_vecs)

# part 2 of the lab
def evolve_embedding(word: str, contexts: Dict[str, Counter], 
                    embeddings: np.ndarray, word_to_idx: Dict[str, int],
                    vocab_list: List[str], stats_dict: Dict[str, float],
                    anchors: Dict[str, List[str]], config: Dict) -> np.ndarray:
    """
    Evolve a single word embedding using (1+λ) Evolution Strategy.
    
    Args:
        word: Target word to insert
        contexts: Context word counts for all target words
        embeddings: Existing embedding matrix
        word_to_idx: Word to index mapping
        vocab_list: List of vocabulary words
        stats_dict: Embedding statistics
        anchors: Anchor words for semantic guidance
        config: Configuration dictionary
    
    Returns:
        Optimized embedding vector
    """
    print(f"\n  Evolving: '{word}'", end='')
    
    dim = embeddings.shape[1]
    mutation_sigma = config['ga_mutation_factor'] * stats_dict['global_std']
    
    # Initialize
    best_vec = initialize_embedding(word, contexts, embeddings, word_to_idx)
    
    # Precompute vectors
    ctx_vecs, ctx_weights, neg_vecs, anchor_vecs = precompute_fitness_vectors(
        word, contexts, embeddings, word_to_idx, vocab_list, anchors
    )
    
    # Initial fitness
    best_fit = compute_fitness(best_vec, word, ctx_vecs, ctx_weights, neg_vecs, 
                               anchor_vecs, stats_dict, config['fitness_weights'])
    
    # Evolution loop
    for gen in range(config['ga_generations']):
        # Generate offspring and evaluate
        population = best_vec + np.random.normal(0, mutation_sigma, (config['ga_pop_size'], dim))
        all_candidates = np.vstack([best_vec, population])
        
        fitness_scores = [compute_fitness(vec, word, ctx_vecs, ctx_weights, neg_vecs, 
                                         anchor_vecs, stats_dict, config['fitness_weights'])
                         for vec in all_candidates]
        
        # Select best
        best_idx = np.argmax(fitness_scores)
        best_vec = all_candidates[best_idx].copy()
        best_fit = fitness_scores[best_idx]
        
        if gen % 50 == 0:
            print(f" G{gen}={best_fit:.4f}", end='')
    
    print(f" ✓ Final={best_fit:.4f}")
    return best_vec


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_with_inserted_words(nodes: List[str], embeddings: np.ndarray, 
                                  inserted_words: List[str],
                                  output_file: str = "embeddings_with_inserted.png",
                                  sample_size: int = 500):
    """Create t-SNE visualization highlighting inserted words."""
    print("\nGenerating t-SNE visualization with inserted words...")
    
    num_original = len(nodes) - len(inserted_words)
    inserted_indices = set(range(num_original, len(nodes)))
    
    # Sample: prioritize inserted words + random original
    if len(nodes) > sample_size:
        sample_indices = list(inserted_indices) + list(np.random.choice(
            num_original, min(sample_size - len(inserted_words), num_original), replace=False))
    else:
        sample_indices = list(range(len(nodes)))
    
    selected_embeddings = embeddings[sample_indices]
    selected_nodes = [nodes[i] for i in sample_indices]
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sample_indices)-1))
    projection = tsne.fit_transform(selected_embeddings)
    
    # Plot
    plt.figure(figsize=(14, 14))
    
    for i in range(len(projection)):
        is_inserted = sample_indices[i] in inserted_indices
        plt.scatter(projection[i, 0], projection[i, 1], 
                   s=200 if is_inserted else 40,
                   alpha=1.0 if is_inserted else 0.6,
                   c='red' if is_inserted else 'steelblue')
        plt.annotate(selected_nodes[i], (projection[i, 0], projection[i, 1]), 
                    fontsize=11 if is_inserted else 9,
                    alpha=1.0 if is_inserted else 0.8,
                    fontweight='bold' if is_inserted else 'normal')
    
    plt.title(f"t-SNE Visualization: {len(sample_indices)} Words "
              f"({sum(1 for i in sample_indices if i in inserted_indices)} Inserted)",
              fontsize=14, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved t-SNE to {output_file}")
    plt.show()


def run_sanity_checks(model: torch.nn.Module, embeddings: np.ndarray, 
                     nodes: List[str], word_to_idx: Dict[str, int]):
    """Run comprehensive sanity checks on loaded model and embeddings."""
    print("\n" + "="*70)
    print("SANITY CHECKS")
    print("="*70)
    
    print(f"\n1. Model Configuration:")
    print(f"   Training mode: {model.training}")
    print(f"   Device: {next(model.parameters()).device}")
    
    print(f"\n2. Embedding Quality:")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Mean: {embeddings.mean():.6f}, Std: {embeddings.std():.6f}")
    print(f"   Min: {embeddings.min():.6f}, Max: {embeddings.max():.6f}")
    print(f"   Contains NaN: {np.isnan(embeddings).any()}, Contains Inf: {np.isinf(embeddings).any()}")
    
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\n3. Embedding Norms:")
    print(f"   Mean: {norms.mean():.4f}, Std: {norms.std():.4f}")
    print(f"   Range: [{norms.min():.4f}, {norms.max():.4f}]")
    
    print(f"\n4. Vocabulary Test:")
    for test_word in ['man', 'woman', 'dog', 'car', 'blue']:
        if test_word in word_to_idx:
            word_idx = word_to_idx[test_word]
            print(f"   '{test_word:10s}' → idx={word_idx:4d}, norm={np.linalg.norm(embeddings[word_idx]):.4f}")
            similar = find_similar_words(test_word, nodes, embeddings, top_k=5)
            if similar:
                print(f"      Similar: {', '.join([f'{w}({s:.3f})' for w, s in similar])}")
    
    print("\n" + "="*70)
    print("✓ SANITY CHECKS COMPLETE")
    print("="*70)



"""
Don't forget to add tests!!!!
"""

class TestEmbeddingInit(unittest.TestCase):
    # If a target word does not have context words, then its initial embedding is the mean.
    def test_no_context_returns_mean(self):
        embeddings = np.random.rand(10, 5)
        word_to_idx = {f"w{i}": i for i in range(10)}
        contexts = {}
        vec = initialize_embedding("unknown_word", contexts, embeddings, word_to_idx)
        np.testing.assert_array_almost_equal(vec, embeddings.mean(axis=0))

    # If a target word does have context words, then its initial embedding is the weighted average of the embeddings of those context words.
    def test_with_context_weighted_average(self):
        embeddings = np.arange(20).reshape(4,5).astype(float)
        word_to_idx = {f"w{i}": i for i in range(4)}
        contexts = {"target": Counter({"w0": 2, "w1": 1})}
        vec = initialize_embedding("target", contexts, embeddings, word_to_idx)
        expected = (2/3)*embeddings[0] + (1/3)*embeddings[1]
        np.testing.assert_array_almost_equal(vec, expected)

class TestFitness(unittest.TestCase):
    # Test fitness values are between 0 and 1
    def test_fitness_range(self):
        vec = np.random.rand(5)
        ctx_vecs = np.random.rand(3,5)
        ctx_weights = np.array([1,2,1])
        neg_vecs = np.random.rand(2,5)
        anchor_vecs = np.random.rand(2,5)
        stats_dict = {'mean_norm':1.0, 'std_norm':0.5, 'global_std':0.5}
        weights = {'corpus':0.5, 'norm':0.3, 'anchor':0.2}
        fitness = compute_fitness(vec, "test", ctx_vecs, ctx_weights, neg_vecs, anchor_vecs, stats_dict, weights)
        self.assertGreaterEqual(fitness, 0.0)
        self.assertLessEqual(fitness, 1.0)

class TestContextExtraction(unittest.TestCase):
    def test_extract_word_contexts(self):
        text = "a quick brown fox jumps over the lazy dog"
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            f.write(text)
            fname = f.name
        contexts = extract_word_contexts(fname, ["fox"], set(text.split()), window=2)
        os.remove(fname)
        self.assertIn("fox", contexts)
        self.assertEqual(contexts["fox"]["brown"], 1)
        self.assertEqual(contexts["fox"]["jumps"], 1)

class TestPrecomputeVectors(unittest.TestCase):
    def test_shapes(self):
        embeddings = np.random.rand(10, 5)
        word_to_idx = {f"w{i}": i for i in range(10)}
        vocab_list = [f"w{i}" for i in range(10)]
        contexts = {"w0": Counter({"w1":2, "w2":1})}
        anchors = {"w0": ["w3", "w4"]}
        ctx_vecs, ctx_weights, neg_vecs, anchor_vecs = precompute_fitness_vectors(
            "w0", contexts, embeddings, word_to_idx, vocab_list, anchors, num_negatives=3
        )
        self.assertEqual(ctx_vecs.shape[1], 5)
        self.assertEqual(neg_vecs.shape, (3, 5))
        self.assertEqual(anchor_vecs.shape[1], 5)

# class TestEvolveEmbedding(unittest.TestCase):
#     def test_smoke_run(self):
#         embeddings = np.random.rand(10, 5)
#         word_to_idx = {f"w{i}": i for i in range(10)}
#         vocab_list = [f"w{i}" for i in range(10)]
#         contexts = {"w0": Counter({"w1":2, "w2":1})}
#         anchors = {"w0": ["w3", "w4"]}
#         stats_dict = {'mean_norm':1.0, 'std_norm':0.5, 'global_std':0.5}
#         config = {'ga_mutation_factor':0.1, 'ga_generations':2, 'ga_pop_size':3, 
#                   'fitness_weights': {'corpus':0.5,'norm':0.3,'anchor':0.2}}
#         vec = evolve_embedding("w0", contexts, embeddings, word_to_idx, vocab_list, stats_dict, anchors, config)
#         self.assertEqual(vec.shape[0], embeddings.shape[1])


# ------------------------------
# RUN TESTS
# ------------------------------
def run_tests():
    """Run all unit tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddingInit))
    suite.addTests(loader.loadTestsFromTestCase(TestFitness))
    suite.addTests(loader.loadTestsFromTestCase(TestContextExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestPrecomputeVectors))
    suite.addTests(loader.loadTestsFromTestCase(TestEvolveEmbedding))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)    
