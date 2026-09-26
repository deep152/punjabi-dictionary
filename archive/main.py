import nltk
from nltk.corpus import wordnet as wn
import pandas as pd
import re as r
from typing import List, Set, Dict
from nltk.corpus import stopwords

#   functions

# add pos
def get_synonyms(definition):
    stop_words = set(stopwords.words('english')) # get English stopwords
    words = definition.split()
   
    # remove stopwords
    filtered_words = []
    for word in words:
        if word not in stop_words:
            filtered_words.append(word)
    words = filtered_words

    synonyms = set()
    for word in words:
        synsets = wn.synsets(word)
        for synset in synsets:
            for lemma in synset.lemmas():
                synonyms.add(lemma.name())
    #   found this last part on SO, basic code to find synonym using lemmas
    return synonyms

def find_possible_synonyms(df, synonyms):
    possible_synonyms = []

    for synonym in synonyms:
        #  check if any synonyms are in definitions 
        matching_rows = df[df['definition'].str.contains(synonym, case=False)] #  makes it so capital/lowercase does not matter
        if not matching_rows.empty:        
            gurmukhi_words = matching_rows['word'].values
            
            #  add these words to list of possible synonyms
            possible_synonyms.extend(gurmukhi_words)
    return list(set(possible_synonyms))[:5]

# not called anywhere yet
def clean_definition(text):
    text = r.sub(r'\([^)]*\)', '', text) # removing content within brackets
    text = text.replace(',', ';') # making all commas into colons to standardize one, not that neccesary
    text = ' '.join(text.split()) # removing white space
    return text.lower()


def main():
    df = pd.read_csv('punjabi_university_dictionary.csv')
    gurmukhi_word = input("Enter a gurmukhi word: ")    # test word
    row = df[df['word'] == gurmukhi_word]   # word column in csv file
    
 # if word exists in dictionary extract definition
    if not row.empty:
        original_definition = row['definition'].values[0]    # extract original definition
        original_pos = row['pos'].values[0]  # extract original POS

        print(f"Definition of {gurmukhi_word}: {original_definition}")

        # Clean the definition before processing
        cleaned_definition = clean_definition(original_definition)

        synonyms = get_synonyms(cleaned_definition)
        possible_synonyms = find_possible_synonyms(df, synonyms, original_pos)

        print(f"\nSynonyms of {gurmukhi_word}:")
        for i, synonym in enumerate(possible_synonyms, 1):
            print(f"Syn {i}: {synonym}")
    else:
        print(f"The word '{gurmukhi_word}' was not found in the dictionary.")

if __name__ == "__main__":
    main()

'''
random notes

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


When finding synonyms it should probably match synonyms that are the same POS

How would the scoring system work; still kind of confused in this

error handling stuff to add

https://stackoverflow.com/questions/5486337/how-to-remove-stop-words-using-nltk-or-python
https://stackoverflow.com/questions/640001/how-can-i-remove-text-within-parentheses-with-a-regex

WordNet only contains "open-class words": nouns, verbs, adjectives, and adverbs. Thus, excluded words include determiners, prepositions, pronouns, conjunctions, and particles.
'''
