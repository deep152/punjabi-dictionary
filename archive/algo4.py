import nltk
from nltk.corpus import wordnet as wn
import pandas as pd
import re

# Download required NLTK data
nltk.download('wordnet', quiet=True)

def load_dictionary(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

def is_gurmukhi(word):
    # Basic check for Gurmukhi script (can be improved)
    return all('\u0A00' <= char <= '\u0A7F' for char in word)

def find_word_definition(df, word):
    return df[df['word'].str.lower() == word.lower()]

def get_synonyms(definition):
    words = definition.split()
    synonyms = set()
    for word in words:
        for synset in wn.synsets(word):
            for lemma in synset.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
    return synonyms

def find_possible_synonyms(df, synonyms, max_synonyms=5):
    possible_synonyms = []
    for synonym in synonyms:
        matching_rows = df[df['definition'].str.contains(r'\b' + re.escape(synonym) + r'\b', case=False, regex=True)]
        if not matching_rows.empty:
            possible_synonyms.extend(matching_rows['word'].values)
            if len(possible_synonyms) >= max_synonyms:
                break
    return list(set(possible_synonyms))[:max_synonyms]

def main():
    df = load_dictionary('punjabi_university_dictionary.csv')
    if df is None:
        return

    while True:
        gurmukhi_word = input("Enter a Punjabi word in Gurmukhi script (or 'q' to quit): ")
        if gurmukhi_word.lower() == 'q':
            break

        if not is_gurmukhi(gurmukhi_word):
            print("Please enter a word in Gurmukhi script.")
            continue

        row = find_word_definition(df, gurmukhi_word)

        if not row.empty:
            definition = row['definition'].values[0]
            print(f"Definition of {gurmukhi_word}: {definition}")

            synonyms = get_synonyms(definition)
            possible_synonyms = find_possible_synonyms(df, synonyms)

            if possible_synonyms:
                print(f"Possible Punjabi synonyms of {gurmukhi_word}: {', '.join(possible_synonyms)}")
            else:
                print(f"No synonyms found for {gurmukhi_word}")
        else:
            print(f"The word '{gurmukhi_word}' was not found in the dictionary.")

if __name__ == "__main__":
    main()