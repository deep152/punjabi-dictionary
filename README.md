# Punjabi Dictionary Tools

My contributions to [The Punjabi App] — a free Punjabi dictionary and language-learning tool. I built this starting in high school together with a family member: I mainly did the dictionary-processing backend, while they handled other parts, front-end, hosting and deployment. Code written 2022–present. However was never cleanly and consistently pushed here.

This repo reflects the current cleaned-up state of my contributions. Additional scripts and iterations from earlier versions of this project exist locally and will be added over time.

## What's in this repo

The existing dictionary tool needed more work to make it comprehensive and optimized for learning. This repo covers the backend scripts I built toward that:

| Script | What it does |
|---|---|
| `synonyms.py` | Suggests related words for a Gurmukhi dictionary entry by comparing English definitions with TF-IDF cosine similarity, filtered to matching part-of-speech |
| `word_difficulty.py` | Scores how "easy" a word is using word frequency (against a 500-common-word list) and word length |
| `fuzzy_search.py` | Typo-tolerant lookup using `difflib`, falls back to close matches when an exact word isn't found |
| `autocomplete_search.py` | Prefix-based search, ranked so more common words surface first |
| `word_of_day.py` | Picks a random "easy" word not previously shown, tracked via a local history file |

All scripts run against `punjabi_university_dictionary.csv` (word, Shahmukhi transliteration, part of speech, English definition, source) and, where relevant, `500_common_punjabi_words.csv`.

## Sample output

[Runs for script 1](other/sample_runs.md)

## Original planning notes

Some of the original notes from when I started one main function for the dictionary, kept because they're honestly a more accurate record of the actual thought process:

> **Task:** Synonyms and antonyms — generate a list of synonyms and antonyms for each word entry, taking the CSV as input and producing a list per word (maybe along with a score).
>
> **Initial plan:**
> - Parse the CSV (pandas)
> - Process definitions: extract and analyze the English definitions for each Punjabi word
> - Use NLTK's WordNet to find synonyms/antonyms for the English words in the definitions
> - Map back to the original Punjabi words
> - Approach could change — this was just laying a starting point
>
> **Things to take care of:**
> - Comma and semicolon splits in the definition column
> - Synonyms/antonyms need to match part of speech
> - Finish `clean_definition`, `split_definition`
>
> **Things to add:**
> - Basic error handling — `isGurmukhi()`, `getWordnetPOS()`, etc.
> - A scoring function to test accuracy
>
> **Open questions:**
> - Polysemy (a word meaning several different things) — how does this factor in?
> - Looked into [l3cube-pune/punjabi-sentence-bert-nli](https://huggingface.co/l3cube-pune/punjabi-sentence-bert-nli) as a possible NLP-based similarity approach instead of pure WordNet/TF-IDF

## Known limitations

- **POS handling is incomplete.** `map_pos()` correctly handles noun/verb/adjective/adverb/conjunct verb, but phrases, interjections, and pronouns all fall through to a shared `None` category, so they can get cross-matched with each other in the synonym finder. Low impact for common vocabulary, more noticeable at full dictionary scale (~300k entries).
- **TF-IDF is refit per query.** `synonyms.py` rebuilds the vectorizer against the whole corpus on every lookup instead of precomputing it once — fine for a demo, not for production at scale.
- **Autocomplete and fuzzy search are linear scans**, not a trie or indexed structure — reasonable for a few thousand rows, not for the full dataset.
- **Word-of-the-day isn't date-locked** — it picks on every run rather than deterministically per calendar day, and the "easy word" heuristic is just length + definition length + common-word lookup rather than real difficulty data.
- None of the scripts handle the Shahmukhi transliteration column yet — only the Gurmukhi `word` column is searched.

## Requirements

```
pandas
scikit-learn
nltk
```

NLTK's `wordnet` and `stopwords` corpora need to be downloaded once:
```python
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
```