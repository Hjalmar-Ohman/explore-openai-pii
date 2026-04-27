#!/usr/bin/env python3
"""
Interactive PII detection demo.

Usage:
    OPF_MOE_TRITON=0 python demo.py
"""
import math
import os
os.environ.setdefault("OPF_MOE_TRITON", "0")

import torch
import torch.nn.functional as F
from opf._api import OPF
from opf._core.runtime import DetectedSpan
from opf._core.sequence_labeling import ExampleAggregation, TokenizedExample, example_to_windows
from opf._core.spans import (
    decode_text_with_offsets,
    discard_overlapping_spans_by_label,
    labels_to_spans,
    token_spans_to_char_spans,
    trim_char_spans_whitespace,
)

RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"
CYAN = "\033[96m"
DIM = "\033[2m"

LABEL_COLORS = {
    "private_person":  "\033[95m",  # magenta
    "private_email":   "\033[94m",  # blue
    "private_phone":   "\033[93m",  # yellow
    "private_address": "\033[92m",  # green
    "private_date":    "\033[96m",  # cyan
    "private_url":     "\033[34m",  # dark blue
    "account_number":  "\033[91m",  # red
}

# Non-PII tokens whose PII probability exceeds this are shown as near-misses.
NEG_THRESHOLD = 0.15


def highlight(text: str, spans: list) -> str:
    result = []
    prev = 0
    for span in sorted(spans, key=lambda s: s.start):
        result.append(text[prev:span.start])
        color = LABEL_COLORS.get(span.label, RED)
        result.append(f"{color}{BOLD}{text[span.start:span.end]}{RESET}{DIM}[{span.label}]{RESET}")
        prev = span.end
    result.append(text[prev:])
    return "".join(result)


def _predict_with_confidence(opf_instance, text, *, neg_threshold=NEG_THRESHOLD):
    """Single inference pass returning (spans_with_confidence, near_miss_groups).

    spans_with_confidence: list of (DetectedSpan, float)
        confidence = mean(1 - p_background) across the span's tokens
    near_miss_groups: list of (phrase_text, avg_pii_prob)
        consecutive non-PII tokens whose PII probability >= neg_threshold
    """
    runtime, decoder = opf_instance.get_prediction_components()

    token_ids = tuple(int(t) for t in runtime.encoding.encode(text, allowed_special="all"))
    if not token_ids:
        return [], []

    bg = int(runtime.label_info.background_token_label)
    example = TokenizedExample(
        tokens=token_ids,
        labels=tuple(bg for _ in token_ids),
        example_id="demo",
        text=text,
    )
    agg = ExampleAggregation(logprob_logsumexp=[], counts=[], labels=[], token_ids=[])

    with torch.inference_mode():
        for window in example_to_windows(example, runtime.n_ctx):
            if not window.tokens:
                continue
            wt = torch.tensor([list(window.tokens)], device=runtime.device, dtype=torch.int32)
            am = torch.ones_like(wt, dtype=torch.bool)
            logits = runtime.model(wt, attention_mask=am)
            lp = F.log_softmax(logits.float(), dim=-1)[0].cpu()
            for pos, valid in enumerate(window.mask):
                if not bool(valid):
                    continue
                tidx = int(window.offsets[pos])
                if tidx < 0:
                    continue
                agg.ensure_capacity(tidx)
                sv = lp[pos]
                ex = agg.logprob_logsumexp[tidx]
                agg.logprob_logsumexp[tidx] = sv.clone() if ex is None else torch.logaddexp(ex, sv)
                agg.counts[tidx] += 1
                agg.record_token_id(tidx, int(window.tokens[pos]), "demo")
                agg.length = max(agg.length, tidx + 1)

    valid_positions, valid_logprobs = [], []
    for tidx in range(agg.length):
        if tidx >= len(agg.logprob_logsumexp):
            continue
        ss, cnt = agg.logprob_logsumexp[tidx], agg.counts[tidx]
        if ss is None or cnt <= 0:
            continue
        valid_positions.append(tidx)
        valid_logprobs.append(ss - math.log(float(cnt)))

    if not valid_logprobs:
        return [], []

    stacked = torch.stack(valid_logprobs, dim=0)  # [seq_len, num_classes]
    bg_probs = stacked.exp()[:, bg]               # background probability per token

    # Decode labels to get span boundaries
    if decoder is not None:
        decoded_labels = decoder.decode(stacked)
        if len(decoded_labels) != len(valid_positions):
            decoded_labels = stacked.argmax(dim=1).tolist()
    else:
        decoded_labels = stacked.argmax(dim=1).tolist()

    labels_by_idx = {tidx: int(lbl) for tidx, lbl in zip(valid_positions, decoded_labels)}
    decoded_text, char_starts, char_ends = decode_text_with_offsets(token_ids, runtime.encoding)
    source_text = decoded_text if decoded_text != text else text

    token_spans = labels_to_spans(labels_by_idx, runtime.label_info)
    char_spans = token_spans_to_char_spans(token_spans, char_starts, char_ends)
    if runtime.trim_span_whitespace:
        char_spans = trim_char_spans_whitespace(char_spans, source_text)
    if runtime.discard_overlapping_predicted_spans:
        char_spans = discard_overlapping_spans_by_label(char_spans)

    # Non-overlapping, left-to-right selection (mirrors _select_non_overlapping_spans)
    raw_sorted = sorted(char_spans, key=lambda s: (s[1], -(s[2] - s[1]), s[0]))
    kept_char_spans, cursor = [], 0
    for label_idx, start, end in raw_sorted:
        if start < cursor or end <= start or not (0 <= start < end <= len(source_text)):
            continue
        kept_char_spans.append((label_idx, start, end))
        cursor = end

    # Build DetectedSpans and compute confidence
    detected_seq_positions: set[int] = set()
    spans_with_conf: list[tuple[DetectedSpan, float]] = []

    for label_idx, start, end in kept_char_spans:
        if 0 <= int(label_idx) < len(runtime.label_info.span_class_names):
            label = str(runtime.label_info.span_class_names[int(label_idx)])
        else:
            label = f"label_{label_idx}"
        span = DetectedSpan(
            label=label,
            start=int(start),
            end=int(end),
            text=source_text[start:end],
            placeholder=f"<{label.upper()}>",
        )
        indices = [
            j for j, tidx in enumerate(valid_positions)
            if tidx < len(char_starts) and char_starts[tidx] < end and char_ends[tidx] > start
        ]
        detected_seq_positions.update(indices)
        conf = float(1.0 - bg_probs[indices].mean().item()) if indices else float("nan")
        spans_with_conf.append((span, conf))

    # Near-misses: non-PII tokens with PII probability >= neg_threshold
    raw_near: list[tuple[int, int, float]] = []
    for j, tidx in enumerate(valid_positions):
        if j in detected_seq_positions or tidx >= len(char_starts):
            continue
        pii_prob = float(1.0 - bg_probs[j].item())
        if pii_prob >= neg_threshold:
            cs, ce = char_starts[tidx], char_ends[tidx]
            tok_text = source_text[cs:ce]
            if tok_text.strip():
                raw_near.append((cs, ce, pii_prob))

    # Group adjacent near-miss tokens into phrases
    near_miss_groups: list[tuple[str, float]] = []
    if raw_near:
        raw_near.sort(key=lambda x: x[0])
        g_start, g_end, g_probs = raw_near[0][:2], raw_near[0][1], [raw_near[0][2]]
        g_start = raw_near[0][0]
        g_end = raw_near[0][1]
        for cs, ce, prob in raw_near[1:]:
            if cs <= g_end:
                g_end = max(g_end, ce)
                g_probs.append(prob)
            else:
                phrase = source_text[g_start:g_end]
                if phrase.strip():
                    near_miss_groups.append((phrase, sum(g_probs) / len(g_probs)))
                g_start, g_end, g_probs = cs, ce, [prob]
        phrase = source_text[g_start:g_end]
        if phrase.strip():
            near_miss_groups.append((phrase, sum(g_probs) / len(g_probs)))

    return spans_with_conf, near_miss_groups


def main() -> None:
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"{CYAN}Loading openai/privacy-filter on {device}...{RESET}")
    opf = OPF(device=device)
    print(f"{CYAN}Ready. Type text to scan for PII. Ctrl+C or empty line to quit.{RESET}\n")

    while True:
        try:
            text = input(f"{BOLD}> {RESET}")
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not text.strip():
            break

        spans_with_conf, near_misses = _predict_with_confidence(opf, text)

        if not spans_with_conf and not near_misses:
            print(f"  {DIM}No PII detected.{RESET}\n")
            continue

        detected_spans = [span for span, _ in spans_with_conf]
        if detected_spans:
            print(f"\n  {highlight(text, detected_spans)}\n")
            for span, conf in sorted(spans_with_conf, key=lambda sc: sc[0].start):
                color = LABEL_COLORS.get(span.label, RED)
                conf_str = f"{conf:.1%}" if math.isfinite(conf) else "n/a"
                print(f"  {color}{span.label:20}{RESET} {repr(span.text):<30} conf: {conf_str}")

        if near_misses:
            print(f"\n  {DIM}Near-misses (PII prob >= {NEG_THRESHOLD:.0%}):{RESET}")
            for phrase, avg_prob in near_misses:
                print(f"  {DIM}{repr(phrase):<34} {avg_prob:.1%}{RESET}")

        print()


if __name__ == "__main__":
    main()
