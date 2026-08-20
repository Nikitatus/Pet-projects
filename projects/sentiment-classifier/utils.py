from typing import Iterable
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score

def calculate_token_frequencies(vocabulary) -> np.ndarray:
    """Calculate frequencies of tokens in a vocabulary.

    Args:
        vocabulary: A vocabulary that contains tokens, 
                    to calculate frequencies for

    Returns:
        np.ndarray: Array of token frequencies (sums to 1), shape (vocab_size,).
    """
    vocab_size = vocabulary.size()
    token_counts = np.zeros(vocab_size)
    for i in range(vocab_size):
        token = vocabulary.id_to_token[i]
        token_counts[i] = vocabulary.tokens_counter[token]

    token_frequincies = token_counts / token_counts.sum()
    return token_frequincies

def compute_positive_logits(model, words, contexts) -> torch.Tensor:
    """Compute logits for word-context pairs.

    Args:
        model: A Word2Vec model that takes (words, contexts) and returns logits.
        words: Tensor of word IDs.
        contexts: Tensor of context word IDs corresponding to each word.

    Returns:
        torch.Tensor: Logits for positive pairs.
    """
    logits = model(words, contexts)
    return logits

def compute_negative_logits(model, words, negative_sample) -> torch.Tensor:
    """Compute logits for negative word-context pairs using negative sampling.

    Args:
        model: A Word2Vec model that takes (words, contexts) and returns logits.
        words: Tensor of word IDs of shape (batch_size,).
        negative_sample: Tensor of negative context samples, shape (batch_size, num_negatives).

    Returns:
        torch.Tensor: Logits for negative pairs, flattened.
    """
    neg_counts = negative_sample.shape[-1]
    words = words.repeat_interleave(neg_counts)
    negatives = negative_sample.flatten()
    logits = model(words, negatives)
    return logits

def calculate_loss(positive_logits, negative_logits) -> torch.Tensor:
    """Compute the negative sampling loss for Word2Vec model.

    Args:
        positive_logits: Logits from positive word-context pairs.
        negative_logits: Logits from negative word-context pairs.

    Returns:
        torch.Tensor: Scalar tensor representing the mean loss.
    """
    positive_loss = (-1) * F.logsigmoid(positive_logits)
    negative_loss = (-1) * F.logsigmoid((-1) * negative_logits)
    loss = positive_loss.mean() + negative_loss.mean()
    return loss

def plot_train_process(train_loss, validation_loss=None):
    """Plot the training (and optionally validation) loss over epochs.

    Args:
        train_loss: List or array of training loss values.
        validation_loss: Optional list or array of validation loss values.
    """
    fig, axes = plt.subplots(1, 1, figsize=(15, 5))
    axes.set_title('Loss')
    axes.plot(train_loss, label='train')
    if validation_loss is not None:
        axes.plot(train_loss, label='validation')
    axes.legend()
    plt.show()

def texts_to_matrix(texts: Iterable[Iterable[int]], 
                    batch_first: bool = True,
                    pad_token_id: int = 0) -> torch.Tensor:
    """Convert a batch of tokenized texts (lists of token IDs) into a padded tensor matrix.

    Args:
        texts: A batch of texts, each represented as a list of token IDs.
        batch_first: If True (default), the output has shape
            (batch_size, max_seq_len). If False, the output has shape
            (max_seq_len, batch_size).
        pad_token_id: ID of the pad token used to pad the sequence to max_seq_len

    Returns:
        torch.Tensor: A padded tensor of shape
            (batch_size, max_seq_len) if `batch_first=True`,
            else (max_seq_len, batch_size). The padding value is
            `vocabulary.token_to_id[pad_token]`.

    Example:
        >>> pad_token_id
        >>> 0
        >>> texts = [[1, 2, 3], [4, 5], [6]]
        >>> matrix = texts_to_matrix(texts)
        >>> matrix
        tensor([[1, 2, 3],
                [4, 5, 0],
                [6, 0, 0]])
    """
    max_text_length = max(map(len, texts))
    batch_size = len(texts)
    matrix = torch.full(size=(batch_size, max_text_length), fill_value=pad_token_id)
    for i in range(batch_size):
        text = texts[i]
        matrix[i, : len(text)] = torch.tensor(text, dtype=matrix.dtype)
    if not batch_first:
        matrix.transpose(0, 1)
    return matrix

def compute_metrics(true_labels, predicted_logits, average=None):
    """Compute standard classification metrics from model outputs.

    Args:
        true_labels: Array-like of true class labels.
        predicted_logits: Torch tensor of predicted logits for each class.
        average: Averaging method for multi-class precision and recall 
                 ('micro', 'macro', 'weighted', or None).

    Returns:
        dict: Dictionary containing accuracy, balanced accuracy, precision, and recall.
    """
    predicted_labels = np.array(torch.argmax(predicted_logits, dim=1).cpu())

    accuracy = accuracy_score(true_labels, predicted_labels)
    balanced_accuracy = balanced_accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average=average)
    recall = recall_score(true_labels, predicted_labels, average=average)
    metrics = {'accuracy': accuracy,
               'balanced_accuracy': balanced_accuracy,
               'precision': precision,
               'recall': recall}
    return metrics