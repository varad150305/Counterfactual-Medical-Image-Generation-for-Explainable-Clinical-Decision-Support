"""Word <-> CLIP-token index mapping, and an original<->edited caption token
alignment ("refinement mapper") for cross-attention control.

[RECONSTRUCTED — interpreted, not verified from source]. Classic
Prompt-to-Prompt ships a `seq_aligner.py` with `get_word_inds` and
`get_refinement_mapper` that we don't have PRISM's exact copy of (it isn't in
PRISM's repo either — PRISM's missing `utils/edit_utils.py` would have
imported something equivalent). This module reimplements the same documented
concept (Hertz et al. 2022, section on unequal-length prompt editing) using
CLIP's own tokenizer directly rather than the BERT-wordpiece-flavored string
heuristic some public P2P ports use, which is fragile against CLIP's BPE
merges. The alignment algorithm (difflib word-level diff -> per-word token
ranges -> position mapper) is our own construction of the described technique,
not a copy of any specific reference file.
"""
import difflib
from typing import List, Tuple, Union

import torch


def get_word_inds(tokenizer, text: str, word_place: Union[str, int]) -> List[int]:
    """Token indices (1-based, i.e. offset past the BOS token — matching CLIP
    tokenizer output layout [BOS, tok..., EOS, pad...]) that a given word
    (matched by exact string, or by its word position if `word_place` is an
    int) occupies in `text`'s CLIP tokenization.
    """
    words = text.split(" ")
    if isinstance(word_place, str):
        positions = [i for i, w in enumerate(words) if w.strip(",.") == word_place.strip(",.")]
    else:
        positions = [word_place]
    if not positions:
        return []

    prefix_token_counts = [0]
    for i in range(1, len(words) + 1):
        prefix = " ".join(words[:i])
        n_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
        prefix_token_counts.append(n_tokens)

    out = []
    for idx in positions:
        start, end = prefix_token_counts[idx], prefix_token_counts[idx + 1]
        out.extend(range(start + 1, end + 1))  # +1: offset past BOS
    return out


def get_refinement_mapper(
    tokenizer, original_text: str, edited_text: str, max_length: int = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aligns `edited_text`'s tokens back to `original_text`'s tokens.

    Returns:
      mapper: LongTensor (max_length,) — for each edited-caption token
        position, the original-caption token index whose attention pattern
        it should (partially) inherit.
      alphas: FloatTensor (max_length,) — 1.0 at positions whose word is
        shared between both captions (pull attention from the original
        branch, keeping that concept visually anchored), 0.0 at positions
        unique to the edited caption (a genuinely new word — e.g. the
        pathology being added — whose attention should develop freely from
        the edited branch itself, since there's nothing in the original to
        anchor it to).
    """
    max_length = max_length or tokenizer.model_max_length
    orig_words = original_text.split(" ")
    edit_words = edited_text.split(" ")

    # Compare case-insensitively (captions differ in sentence-initial
    # capitalization, e.g. "Chest X-ray..." vs "...chest X-ray...") while
    # still tokenizing/indexing the original-cased text below.
    sm = difflib.SequenceMatcher(a=[w.lower() for w in orig_words], b=[w.lower() for w in edit_words])
    mapper = torch.zeros(max_length, dtype=torch.long)
    alphas = torch.zeros(max_length, dtype=torch.float32)

    # BOS token: always aligned to BOS, fully anchored
    mapper[0] = 0
    alphas[0] = 1.0

    last_orig_word_idx = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for oi, ei in zip(range(i1, i2), range(j1, j2)):
                orig_inds = get_word_inds(tokenizer, original_text, oi)
                edit_inds = get_word_inds(tokenizer, edited_text, ei)
                for k, ind in enumerate(edit_inds):
                    if ind >= max_length:
                        continue
                    mapper[ind] = orig_inds[min(k, len(orig_inds) - 1)] if orig_inds else 0
                    alphas[ind] = 1.0
                if orig_inds:
                    last_orig_word_idx = orig_inds[-1]
        else:
            # replace/insert/delete: edited-side new tokens inherit the
            # nearest preceding original token (anchor), but are NOT forced
            # to match its attention (alpha=0) — they should develop their
            # own pattern.
            for ei in range(j1, j2):
                edit_inds = get_word_inds(tokenizer, edited_text, ei)
                for ind in edit_inds:
                    if ind >= max_length:
                        continue
                    mapper[ind] = last_orig_word_idx
                    alphas[ind] = 0.0

    # EOS + padding positions: map positionally (edited EOS -> original EOS,
    # then pad k -> pad k), fully anchored. Must NOT collapse them all onto
    # one source index: that copies a single token's attention into every
    # padding column, so each attention row sums to far more than 1 and the
    # output is scaled up (this was a real bug here — see
    # tests/test_prism_baseline.py::TestControllerInvariants). With identical
    # prompts this reduces to the identity map, as it must.
    orig_token_count = len(tokenizer.encode(original_text, add_special_tokens=False)) + 2  # +BOS+EOS
    edit_token_count = len(tokenizer.encode(edited_text, add_special_tokens=False)) + 2
    for k, pos in enumerate(range(edit_token_count - 1, max_length)):
        mapper[pos] = min(orig_token_count - 1 + k, max_length - 1)
        alphas[pos] = 1.0

    return mapper, alphas.unsqueeze(-1)


def get_equalizer(tokenizer, text: str, words: List[str], values: List[float], max_length: int = None) -> torch.Tensor:
    """Multiplicative per-token weight vector, 1.0 everywhere except the
    given words' token positions (set to `values`) — implements PRISM's
    `edit_word_weight` / `eq_params`."""
    max_length = max_length or tokenizer.model_max_length
    equalizer = torch.ones(1, max_length)
    if isinstance(words, str):
        words = [words]
    if isinstance(values, (int, float)):
        values = [values] * len(words)
    for word, val in zip(words, values):
        for ind in get_word_inds(tokenizer, text, word):
            if ind < max_length:
                equalizer[0, ind] = val
    return equalizer
