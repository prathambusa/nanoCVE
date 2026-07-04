"""
data/prepare.py — Download NVD CVE data, extract descriptions, and build
a tokenized corpus for nanoCVE pretraining.

Pipeline
--------
1. Fetch CVEs from NVD API 2.0 (services.nvd.nist.gov/rest/json/cves/2.0),
   paginating through all results in batches of 2000.
2. Extract the English description field from each CVE entry.
3. Deduplicate by CVE-ID, strip nulls and uninformative boilerplate.
4. Concatenate into one corpus with a newline separator between entries.
5. Tokenize with both char and BPE tokenizers and save as numpy memmaps.
6. Report corpus statistics.

Rate limits (NVD API 2.0)
--------------------------
  Without API key : 5 requests / 30 seconds  (~180 pages ≈ 18 min)
  With API key    : 50 requests / 30 seconds  (~180 pages ≈ 2 min)
  Set NVD_API_KEY env var to use your key:
    export NVD_API_KEY=your-key-here

Outputs (written to data/cache/)
---------------------------------
  corpus.txt          — raw concatenated descriptions (UTF-8)
  train.txt           — 90% split
  val.txt             — 10% split
  char_vocab.json     — char tokenizer vocabulary
  train_char.bin      — uint16 token array (char-level)
  val_char.bin        — uint16 token array (char-level)
  train_bpe.bin       — uint32 token array (BPE / tiktoken gpt2)
  val_bpe.bin         — uint32 token array (BPE / tiktoken gpt2)

Usage
-----
  python data/prepare.py                # full corpus (~360k CVEs)
  python data/prepare.py --limit 20000  # fast smoke-test
  python data/prepare.py --force        # ignore cache, re-download
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
CACHE = ROOT / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

NVD_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"
PAGE_SIZE = 2000  # max allowed by the API


# ── Download helpers ──────────────────────────────────────────────────────────

def _fetch_page(start_index: int, api_key: str | None) -> dict:
    """Fetch one page of CVEs from NVD API 2.0."""
    params = {"resultsPerPage": PAGE_SIZE, "startIndex": start_index}
    url = f"{NVD_API}?{urllib.parse.urlencode(params)}"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["apiKey"] = api_key

    req = urllib.request.Request(url, headers=headers)
    # NVD can be slow under load; 120s covers observed ~37s response times
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _extract_from_page(page: dict) -> list[tuple[str, str]]:
    """Extract (cve_id, english_description) pairs from one API page."""
    results = []
    for item in page.get("vulnerabilities", []):
        cve = item.get("cve", {})
        cve_id = cve.get("id", "")
        for d in cve.get("descriptions", []):
            if d.get("lang") == "en":
                text = d.get("value", "").strip()
                if text:
                    results.append((cve_id, text))
                break
    return results


def download_all_cves(limit: int | None, api_key: str | None) -> list[tuple[str, str]]:
    """
    Paginate through the NVD API 2.0 and collect all (cve_id, description) pairs.
    Respects rate limits automatically.
    """
    # First request to learn total count. If the key returns 404 (not yet
    # activated), automatically fall back to unauthenticated mode.
    print("  Probing API for total CVE count...", end=" ", flush=True)
    try:
        first_page = _fetch_page(0, api_key)
    except urllib.error.HTTPError as e:
        if e.code == 404 and api_key:
            print(f"\n  API key returned 404 (not yet activated) — falling back to no-key mode")
            api_key = None
            first_page = _fetch_page(0, api_key)
        else:
            raise
    total = first_page["totalResults"]
    print(f"{total:,} CVEs available")

    if limit:
        total = min(total, limit)

    n_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
    # Rate limit: 5 req/30s without key, 50 req/30s with key
    # Sleep between requests to stay safely under the limit.
    sleep_s = 6.5 if not api_key else 0.7

    all_pairs: list[tuple[str, str]] = []

    # Process first page (already fetched)
    pairs = _extract_from_page(first_page)
    all_pairs.extend(pairs)
    print(f"  Page   1/{n_pages}  ({len(all_pairs):,} entries so far)")

    for page_num in range(1, n_pages):
        start = page_num * PAGE_SIZE
        if limit and start >= limit:
            break

        # Polite sleep to respect NVD rate limits
        time.sleep(sleep_s)

        try:
            page = _fetch_page(start, api_key)
            pairs = _extract_from_page(page)
            all_pairs.extend(pairs)
        except Exception as e:
            print(f"\n  WARNING: page {page_num} failed ({e}), retrying in 35s...")
            time.sleep(35)
            try:
                page = _fetch_page(start, api_key)
                pairs = _extract_from_page(page)
                all_pairs.extend(pairs)
            except Exception as e2:
                print(f"  SKIPPING page {page_num} after retry: {e2}")
                continue

        if page_num % 10 == 0 or page_num == n_pages - 1:
            print(f"  Page {page_num+1:>3d}/{n_pages}  ({len(all_pairs):,} entries so far)")

        if limit and len(all_pairs) >= limit:
            all_pairs = all_pairs[:limit]
            print(f"  --limit {limit} reached")
            break

    return all_pairs


# ── Text cleaning ─────────────────────────────────────────────────────────────

_BOILERPLATE = re.compile(
    r"^\*\*\s*RESERVED\s*\*\*|^\*\*\s*REJECT\s*\*\*|^This candidate has been reserved",
    re.IGNORECASE,
)


def _clean(text: str) -> str | None:
    """Return cleaned text, or None if the entry should be dropped."""
    if _BOILERPLATE.match(text):
        return None
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 30:
        return None
    return text


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_corpus(limit: int | None = None, force: bool = False) -> tuple[str, str]:
    """
    Download, clean, and split the CVE corpus.
    Returns (train_text, val_text). Results cached in data/cache/.
    """
    corpus_path = CACHE / "corpus.txt"
    train_path  = CACHE / "train.txt"
    val_path    = CACHE / "val.txt"

    # Cache hit: skip download if files exist and are non-empty
    if not force and corpus_path.exists() and corpus_path.stat().st_size > 1000:
        print("Cache hit — loading corpus from data/cache/ (use --force to re-download)")
        return train_path.read_text(encoding="utf-8"), val_path.read_text(encoding="utf-8")

    api_key = os.environ.get("NVD_API_KEY")
    if api_key:
        print(f"Using NVD API key from NVD_API_KEY env var")
    else:
        print("No NVD_API_KEY set — using unauthenticated rate limit (5 req/30s).")
        print("Expected download time: ~20 min for full corpus.")
        print("Set NVD_API_KEY for ~10x faster download.\n")

    print("=" * 60)
    print("Step 1/4: Downloading CVEs from NVD API 2.0")
    print("=" * 60)

    all_pairs = download_all_cves(limit=limit, api_key=api_key)
    print(f"\nRaw entries collected: {len(all_pairs):,}")

    print("\nStep 2/4: Deduplicating and cleaning")
    seen_ids: set[str] = set()
    descriptions: list[str] = []
    for cve_id, text in all_pairs:
        if cve_id in seen_ids:
            continue
        seen_ids.add(cve_id)
        cleaned = _clean(text)
        if cleaned:
            descriptions.append(cleaned)

    print(f"  After dedup + clean: {len(descriptions):,} descriptions")

    print("\nStep 3/4: Building train/val split (90/10 by document)")
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(descriptions))
    split = int(0.9 * len(descriptions))
    train_docs = [descriptions[i] for i in indices[:split]]
    val_docs   = [descriptions[i] for i in indices[split:]]

    train_text = "\n\n".join(train_docs)
    val_text   = "\n\n".join(val_docs)

    corpus_path.write_text("\n\n".join(descriptions), encoding="utf-8")
    train_path.write_text(train_text, encoding="utf-8")
    val_path.write_text(val_text, encoding="utf-8")
    print(f"  Saved corpus.txt ({corpus_path.stat().st_size / 1e6:.1f} MB)")

    return train_text, val_text


def tokenize_and_save(train_text: str, val_text: str) -> None:
    """Tokenize with both tokenizers and save as numpy memmaps."""
    print("\nStep 4/4: Tokenizing")

    sys.path.insert(0, str(ROOT))
    from tokenizer import CharTokenizer, BPETokenizer

    # ── Char tokenizer ────────────────────────────────────────────────────────
    char_tok = CharTokenizer()
    char_tok.build_vocab(train_text + val_text)
    char_tok.save(CACHE / "char_vocab.json")

    for split, text, fname in [
        ("train", train_text, "train_char.bin"),
        ("val",   val_text,   "val_char.bin"),
    ]:
        ids = char_tok.encode(text)
        np.array(ids, dtype=np.uint16).tofile(CACHE / fname)
        print(f"  char {split}: {len(ids):,} tokens → {fname}")

    # ── BPE tokenizer ─────────────────────────────────────────────────────────
    bpe_tok = BPETokenizer()
    for split, text, fname in [
        ("train", train_text, "train_bpe.bin"),
        ("val",   val_text,   "val_bpe.bin"),
    ]:
        ids = bpe_tok.encode(text)
        np.array(ids, dtype=np.uint32).tofile(CACHE / fname)
        print(f"  bpe  {split}: {len(ids):,} tokens → {fname}")

    # ── Stats ─────────────────────────────────────────────────────────────────
    full_corpus = train_text + val_text
    n_docs  = full_corpus.count("\n\n") + 1
    n_chars = len(full_corpus)
    char_train = np.fromfile(CACHE / "train_char.bin", dtype=np.uint16)
    bpe_train  = np.fromfile(CACHE / "train_bpe.bin",  dtype=np.uint32)

    print("\n" + "=" * 60)
    print("Corpus statistics")
    print("=" * 60)
    print(f"  Documents (total)   : {n_docs:>12,}")
    print(f"  Characters          : {n_chars:>12,}")
    print(f"  Char vocab size     : {char_tok.vocab_size:>12,}")
    print(f"  BPE vocab size      : {bpe_tok.vocab_size:>12,}")
    print(f"  Tokens (char, train): {len(char_train):>12,}")
    print(f"  Tokens (bpe,  train): {len(bpe_train):>12,}")
    ratio = len(char_train) / max(len(bpe_train), 1)
    print(f"  Char/BPE ratio      : {ratio:>12.2f}x  (char seqs are longer)")
    print("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare nanoCVE corpus from NVD API 2.0")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap CVE count (e.g. 20000 for a fast smoke-test)")
    parser.add_argument("--force", action="store_true",
                        help="Ignore cache and re-download everything")
    args = parser.parse_args()

    train_text, val_text = build_corpus(limit=args.limit, force=args.force)
    tokenize_and_save(train_text, val_text)
    print("\nDone. Ready to train — run: make train-bpe")


if __name__ == "__main__":
    main()
