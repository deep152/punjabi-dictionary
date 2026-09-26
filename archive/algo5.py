import nltk
from nltk.corpus import wordnet as wn
import pandas as pd

#   functions

#   parsing through csv using pandas
def load_dictionary(file_path):
    return pd.read_csv(file_path)

def find_word_definition(df, word):
    return df[df['word'] == word]

#   using wordnet to get synonyms
def get_synonyms(definition):
    #   split definition into seperate words, this part will change
    words_in_definition = definition.split()
    synonyms = set()
    for word in words_in_definition:
        for synonym in wn.synsets(word):
            for lemma in synonym.lemmas():
                synonyms.add(lemma.name())
    return synonyms
#   found this on SO, basic code to find synonym

def find_possible_synonyms(df, synonyms):
    possible_synonyms = []

    for synonym in synonyms:
        #   check if any synonyms are in definitions and return the corresponding gurmukhi words
        matching_rows = df[df['definition'].str.contains(synonym, case=False)]
        if not matching_rows.empty:
            possible_synonyms.extend(matching_rows['word'].values)
            if len(possible_synonyms) >= 5: #   so it doesnt return a lot of words
                break
    return list(set(possible_synonyms))[:5]

# functions to do

def clean_definition(text):
    '''
    remove extra spaces
    standardize semicolons and commas
    remove brackets and whatevers inside (if neccesary)
    '''

def split_definition(definition):
    '''
    split by commas and semicolons
    re library maybe
    '''

def main():
    df = load_dictionary('punjabi_university_dictionary.csv')
    gurmukhi_word = input("Enter a gurmukhi word: ")    # test word
    row = find_word_definition(df, gurmukhi_word)   # word column in csv file
    
    # if word exists in dictionary extract definition
    if not row.empty:
        definition = row['definition'].values[0]    # extract English definition
        print(f"Definition of {gurmukhi_word}: {definition}")

        synonyms = get_synonyms(definition)
        possible_synonyms = find_possible_synonyms(df, synonyms)

        print(f"Synonyms of {gurmukhi_word}:")
        for i, synonym in enumerate(possible_synonyms, 1):
            print(f"Syn {i}: {synonym}")
    else:
        print(f"The word '{gurmukhi_word}' was not found in the dictionary.")

if __name__ == "__main__":
    main()


'''
notes

Parse the CSV: Use Python's CSV module or pandas to read the dictionary file.
Process Definitions: Extract and analyze the English definitions for each Punjabi word.
Use WordNet: Use NLTK's WordNet to find synonyms and antonyms for the English words in the definitions.
Map Back to Punjabi: Associate the found synonyms and antonyms with the original Punjabi words.

right now its just a simple algorithm following the things above. The approach will be changed as we like, just laying a ground for now
Wordnet has its limatitions like you said

When finding synonyms it should probably match synonyms that are the same POS
How would the scoring system work

polysemy- when a word or phrase means many different things

for conjunct phrases like ਉਸਤਤ ਕਰਨੀ how would the approach be 
I haven't tested a lot of those words, just simple words like joyful, dawn, etc and got some possible synonyms

random things I came across/links that might help

https://huggingface.co/l3cube-pune/punjabi-sentence-bert-nli
https://baotramduong.medium.com/nlp-semantic-similarity-identifying-synonyms-in-a-large-corpus-of-words-8b8edc9ce1f9

'''
def find_possible_synonyms(df, synonyms):
    possible_synonyms = []
    
    for synonym in synonyms:
        # Check if any synonyms are in definitions
        matching_rows = df[df['definition'].str.contains(synonym, case=False)]
        
        # If matches found, process them
        if not matching_rows.empty:
            # Extract the corresponding Gurmukhi words
            gurmukhi_words = matching_rows['word'].values
            
            # Add these words to our list of possible synonyms
            possible_synonyms.extend(gurmukhi_words)
            
            # Check if we have enough synonyms
            if len(possible_synonyms) >= 5:
                # Stop searching if we have 5 or more synonyms
                break
    
    # Remove duplicates from the list of synonyms
    unique_synonyms = list(set(possible_synonyms))
    
    # Return up to 5 unique synonyms
    return unique_synonyms[:5]