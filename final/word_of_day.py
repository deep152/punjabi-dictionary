import pandas as pd
import random
import os

HISTORY_FILE = 'word_of_the_day_history.csv'  # keeps track of what we already used


def load_used_words():
    """loads previously used words so we dont repeat them"""
    if not os.path.exists(HISTORY_FILE):
        return set()
    history_df = pd.read_csv(HISTORY_FILE)
    return set(history_df['word'].tolist())


def save_used_word(word, definition, pos):
    """appends today's pick to the history file"""
    new_row = pd.DataFrame([{'word': word, 'definition': definition, 'pos': pos}])
    if os.path.exists(HISTORY_FILE):
        new_row.to_csv(HISTORY_FILE, mode='a', header=False, index=False)
    else:
        new_row.to_csv(HISTORY_FILE, mode='w', header=True, index=False)


def is_easy_word(word, definition, common_words):
    """rough check for whether a word counts as 'easy' enough for word of the day
    not perfect but good enough for now
    """
    if len(word) > 6:  # longer gurmukhi words tend to be more complex
        return False
    if len(definition) > 60:  # long definitions usually mean multiple senses / harder word
        return False
    if word in common_words:  # bonus points if its already a known common word
        return True
    return len(word) <= 4  # short words are usually simple even if not in common list


def pick_word_of_the_day(df, common_words):
    """picks a random easy word that hasn't been used before"""
    used_words = load_used_words()

    # filter down to easy candidates first, way faster than checking one at a time
    candidates = []
    for idx, row in df.iterrows():
        word = row['word']
        if word in used_words:
            continue
        if is_easy_word(word, str(row['definition']), common_words):
            candidates.append(row)

    if not candidates:
        print("No new easy words left! (or ran out, maybe reset history file)")
        return None

    pick = random.choice(candidates)
    return pick


def main():
    try:
        df = pd.read_csv('punjabi_university_dictionary.csv')
    except FileNotFoundError:
        print("Error: 'punjabi_university_dictionary.csv' not found.")
        return

    try:
        common_words = pd.read_csv('500_common_punjabi_words.csv')['Word'].tolist()[1:]
    except FileNotFoundError:
        print("Warning: '500_common_punjabi_words.csv' not found, skipping commonness check.")
        common_words = []

    pick = pick_word_of_the_day(df, common_words)

    if pick is not None:
        word = pick['word']
        pos = pick['pos']
        definition = pick['definition']

        print("Word of the Day")
        print(word)
        print(f"{pos}")
        print(f"{definition}")

        save_used_word(word, definition, pos)


if __name__ == "__main__":
    main()

"""
is_easy_word is pretty simple for now, just length + definition length + common word check
future version 
- use difficulty scorer (500 common words freq one) script instead
- history file is just for this
"""