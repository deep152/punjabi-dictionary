import pandas as pd
from collections import defaultdict
import nltk
from nltk.corpus import wordnet as wn
import ssl

nltk.download('wordnet')

# to bypass certificate verification error
ssl._create_default_https_context = ssl._create_unverified_context

# load csv file
dict = pd.read_csv('punjabi_university_dictionary.csv')

selectWord = input("word? ")

# get synonyms using wordnet
def get_synonyms(selectWord):
    synonyms = set()
    for syn in wn.synsets(selectWord):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms

# find synonyms of a Gurmukhi word by analyzing its English definition
def find_gurmukhi_synonyms(gurmukhi_word):
    # Find the row corresponding to the input Gurmukhi word
    row = dict[dict['word'] == gurmukhi_word]
    if row.empty:
        return f"No definition found for the word {gurmukhi_word}"
    
    # Extract the English definition
    english_definition = row['definition'].iloc[0]
    
    # Split the definition into words (you can improve this by focusing on key nouns or verbs)
    definition_words = english_definition.split()
    
    # Get synonyms for key words in the definition
    synonyms = set()
    for word in definition_words:
        synonyms.update(get_synonyms(word))
    
    # Map synonyms back to Gurmukhi words in the dictionary based on similar definitions
    gurmukhi_synonyms = []
    for _, other_row in dict.iterrows():
        other_definition = other_row['definition']
        for synonym in synonyms:
            if synonym in other_definition:
                gurmukhi_synonyms.append(other_row['word'])
    
    return list(set(gurmukhi_synonyms))  # Remove duplicates

