import pandas as pd


def edit_distance(word1, word2):
    """basic levenshtein distance, doing it manually instead of importing a library
    probably slower than a real library but works fine for single word lookups
    """
    m, n = len(word1), len(word2)
    # dp table, dp[i][j] = edit distance between word1[:i] and word2[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


def get_fuzzy_matches(typo_word, df, max_distance=2, max_results=5):
    """finds words within max_distance edits of the input, closest first"""
    scored = []
    for idx, row in df.iterrows():
        word = row['word']
        # skip words with wildly different lengths, saves time and they wont match anyway
        if abs(len(word) - len(typo_word)) > max_distance:
            continue
        dist = edit_distance(typo_word, word)
        if dist <= max_distance and dist > 0:  # dist 0 means exact match, not a typo
            scored.append((row, dist))

    scored.sort(key=lambda x: x[1])  # closest matches first
    return scored[:max_results]


def truncate_text(text, length=40):
    """Truncates text and adds ellipsis if needed."""
    return text[:length] + "..." if len(text) > length else text


def main():
    try:
        df = pd.read_csv('punjabi_university_dictionary.csv')
    except FileNotFoundError:
        print("Error: 'punjabi_university_dictionary.csv' not found.")
        return

    while True:
        typo_word = input("\nEnter a Gurmukhi word (or type 'exit'): ")
        if typo_word.lower() == 'exit':
            break

        exact = df[df['word'] == typo_word]
        if not exact.empty:
            row = exact.iloc[0]
            print(f"{row['word']} ({row['pos']})")
            print(f"DEF: {truncate_text(row['definition'])}")
            print("=" * 30)
            continue  # found exact match, no need to fuzzy search

        print(f"'{typo_word}' not found exactly, checking for close matches...")
        results = get_fuzzy_matches(typo_word, df)

        if results:
            for row, dist in results:
                print(f"  [{dist} edits] {row['word']} ({row['pos']}) - {truncate_text(row['definition'])}")
            print("=" * 30)
        else:
            print(f"No close matches found for '{typo_word}'.")
            print("=" * 30)


if __name__ == "__main__":
    main()

"""
notes:
- edit_distance is brute force, O(n*m) per word times every row in the df, way too slow at 300k rows
- real version would need something like a BK-tree or just use a library (rapidfuzz) instead
- max_distance=2 is just a guess, could be too loose or too strict depending on word length
- only checks gurmukhi word column, not shahmukhi like all other scripts
"""