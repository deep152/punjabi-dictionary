import pandas as pd
import re as r


def get_word_commonness_score(word, common_words):
    """get commonness of word according to common words list"""
    if word not in common_words:
        return 0.1
    position = common_words.index(word)
    if position < 20:
        return 1.0
    elif position < 50:
        return 0.9
    elif position < 100:
        return 0.8
    elif position < 220:
        return 0.7
    return 0.6


def get_length_penalty(word):
    """longer words tend to be harder, just a rough estimate honestly"""
    length = len(word)
    if length <= 3:
        return 1.0
    elif length <= 6:
        return 0.8
    elif length <= 9:
        return 0.6
    else:
        return 0.4


def calculate_difficulty_score(word, common_words):
    """Combines commonness + length into one difficulty score.
    higher score = easier word vv
    can prob be weighted differently
    """
    commonness_score = get_word_commonness_score(word, common_words)
    length_score = get_length_penalty(word)
    final_score = (commonness_score * 0.7) + (length_score * 0.3)
    return final_score


def label_difficulty(score):
    """turns the number into something readable"""
    if score >= 0.85:
        return "Easy"
    elif score >= 0.6:
        return "Medium"
    elif score >= 0.35:
        return "Hard"
    else:
        return "Very Hard"


def truncate_text(text, length=40):
    """Truncates text and adds ellipsis if needed."""
    return text[:length] + "..." if len(text) > length else text


def main():
    try:
        df = pd.read_csv('punjabi_university_dictionary.csv')
    except FileNotFoundError:
        print("Error: 'punjabi_university_dictionary.csv' not found.")
        return

    try:
        common_words = pd.read_csv('500_common_punjabi_words.csv')['Word'].tolist()[1:]
    except FileNotFoundError:
        print("Error: '500_common_punjabi_words.csv' not found.")
        return

    while True:
        gurmukhi_word = input("\nEnter a Gurmukhi word (or type 'exit'): ")
        if gurmukhi_word.lower() == 'exit':
            break

        row = df[df['word'] == gurmukhi_word]  # look up the word in csv

        if not row.empty:
            definition = row['definition'].values[0]
            pos = row['pos'].values[0]

            score = calculate_difficulty_score(gurmukhi_word, common_words)
            label = label_difficulty(score)

            print(f"{gurmukhi_word} ({pos})")
            print(f"DEF: {truncate_text(definition)}")
            print(f"Difficulty: {label}  [score: {score:.3f}]")
            print("=" * 30)  # separator
        else:
            print(f"The word '{gurmukhi_word}' was not found in the dictionary.")
            print("=" * 50)


if __name__ == "__main__":
    main()

"""
length penalty is really rough, doesnt account for actual pronunciation difficulty
would be better to score using the sentence corpus frequency instead of just the 500 word list
some words not in common_words at all just default to 0.1 which might be too harsh
"""
