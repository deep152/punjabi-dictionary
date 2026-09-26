import nltk
from nltk.corpus import wordnet as wn
import pandas as pd

df = pd.read_csv('punjabi_university_dictionary.csv')

gurmukhi_word = input("What word? " )   # test word
row = df[df['word'] == gurmukhi_word]   # word column in csv file

# functions to do

def clean_definition(text):
    '''
    remove extra spaces
    standardize semicolons and commas
    remove brackets and their contents (if neccesary)
    '''

def split_definition(definition):
    '''
    split by commas and semicolons
    re library maybe
    '''

# if word exists in dictionary extract definition
if not row.empty:
    definition = row['definition'].values[0]    # extract English definition
    print(f"Definition of {gurmukhi_word}: {definition}")
    
    words_in_definition = definition.split()

    # find synonym for first word in the definition using Wordnet
    synonyms = set()
    for word in words_in_definition:
        for synonym in wn.synsets(word):
            for lemma in synonym.lemmas():
                synonyms.add(lemma.name())
    
    # check if any synonyms are in definitions and return the corresponding Gurmukhi words
    possible_synonyms = []
    for synonym in synonyms:
        matching_rows = df[df['definition'].str.contains(synonym, case=False)]
        if not matching_rows.empty:
            possible_synonyms.extend(matching_rows['word'].values)
            if len(possible_synonyms) >= 5:  
                break

    print(f"Synonyms of {gurmukhi_word}: {possible_synonyms}")
else:
    print(f"The word '{gurmukhi_word}' was not found in the dictionary")

'''
Parse the CSV: Use Python's CSV module or pandas to read the dictionary file.
Process Definitions: Extract and analyze the English definitions for each Punjabi word.
Use WordNet: Use NLTK's WordNet to find synonyms and antonyms for the English words in the definitions.
Map Back to Punjabi: Associate the found synonyms and antonyms with the original Punjabi words.

right now its just a simple algorithm following the things above, approach can be changed just laying a ground
wordnet has its limatitions like you said

when finding synonyms it should probably match synonyms that are the same POS
for conjunct phrases like ਉਸਤਤ ਕਰਨੀ how would the approach be 

I was just searching up random things could something like this maybe help 
https://huggingface.co/l3cube-pune/punjabi-sentence-bert-nli


'''