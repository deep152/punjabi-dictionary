import nltk
from nltk.corpus import wordnet as wn
import pandas as pd
import re as r
from typing import List, Set, Dict
from nltk.corpus import stopwords

#   functions

# maps parts of speech for when findning synonyms
def map_pos(pos):
    if 'noun' in pos:
        return 'NOUN'
    elif 'verb' in pos or 'conjunct verb' in pos:
        return 'VERB'
    elif 'adjective' in pos:
        return 'ADJ'
    elif 'adverb' in pos:
        return 'ADV'
    elif 'pronoun' in pos:
        return 'PRON'
    elif 'phrase' in pos:
        return 'PHRASE'
    elif 'interjection' in pos:
        return 'INTJ'
    else:
        return None

def get_synonyms(definition: str, df: pd.DataFrame, original_pos: str) -> Set[str]:
    stop_words = set(stopwords.words('english'))
    words = definition.lower().split()
    filtered_words = [word for word in words if word not in stop_words]
    
    synonyms = set()
    for i in range(len(filtered_words)):
        for j in range(i+1, min(i+4, len(filtered_words))):
            phrase = ' '.join(filtered_words[i:j])
            matching_rows = df[
                (df['definition'].str.contains(phrase, case=False, regex=False)) &
                (df['pos'].str.contains(map_pos(original_pos), case=False))
            ]
            if not matching_rows.empty:
                synonyms.update(matching_rows['word'].tolist())
    
    return synonyms

def find_possible_synonyms(df, synonyms, original_pos):
    possible_synonyms = []
    # change POS to either of the options
    original_pos_category = map_pos(original_pos)
    for synonym in synonyms:
        #  check if any synonyms are in definitions and matches pos
        matching_rows = df[(df['definition'].str.contains(synonym, case=False)) & 
                           (df['pos'].str.contains(original_pos_category, case=False))]
        if not matching_rows.empty:        
            gurmukhi_words = matching_rows['word'].values
            #  add these words to list of possible synonyms
            possible_synonyms.extend(gurmukhi_words)
    return list(set(possible_synonyms))[:5]

def clean_definition(text):
    text = r.sub(r'\([^)]*\)', '', text) # removing content within brackets
    text = text.replace(',', ';') # making all commas into colons to standardize one, not that neccesary
    text = ' '.join(text.split()) # removing white space
    return text.lower()


def main():
    df = pd.read_csv('punjabi_university_dictionary.csv')
    gurmukhi_word = input("Enter a gurmukhi word: ")
    row = df[df['word'] == gurmukhi_word]

    if not row.empty:
        original_definition = row['definition'].values[0]
        original_pos = row['pos'].values[0]

        print(f"Definition of {gurmukhi_word}: {original_definition}")

        cleaned_definition = clean_definition(original_definition)

        synonyms = get_synonyms(cleaned_definition, df, original_pos)
        possible_synonyms = find_possible_synonyms(df, synonyms, original_pos)

        print(f"\nSynonyms of {gurmukhi_word} ({original_pos}):")
        for i, synonym in enumerate(possible_synonyms, 1):
            print(f"Syn {i}: {synonym}")
    else:
        print(f"The word '{gurmukhi_word}' was not found in the dictionary.")
if __name__ == "__main__":
    main()

'''
random notes

if we want to use pos tagging
WordNet uses these POS codes:
n: noun
v: verb
a: adjective
r: adverb


deleted split_definition, done in find_possible_synonyms function

now removes extra words in definition:
definition = "the act of giving praise to someone"
the program will find synonyms for "the", "of", "to" - waste of time and it could also match unrelated words cus of common stopwords

with stopword removal:

definition = "act giving praise someone"

I think with this it'll be better since it will find synonyms for the meaningful words instead of the, to..  


Some words might not return accurate synonyms for unique religious/cultural terms

How would the scoring system work; still kind of confused on this

error handling stuff to add

https://stackoverflow.com/questions/5486337/how-to-remove-stop-words-using-nltk-or-python
https://stackoverflow.com/questions/640001/how-can-i-remove-text-within-parentheses-with-a-regex

WordNet only contains "open-class words": nouns, verbs, adjectives, and adverbs. Thus, excluded words include determiners, prepositions, pronouns, conjunctions, and particles.
'''
