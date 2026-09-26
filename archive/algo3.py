import nltk
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.wsd import lesk
import pandas as pd
import re

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Load the CSV file
df = pd.read_csv('punjabi_university_dictionary.csv')

def clean_definition(text):
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    text = re.sub(r';', ',', text)  # standardize semicolons to commas
    text = re.sub(r'\([^)]*\)', '', text)  # remove brackets and their contents
    return text.strip()

def split_definition(definition):
    return [word.strip() for word in re.split(r'[,;]', definition) if word.strip()]

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None

def find_synonyms(word, pos=None):
    synonyms = set()
    for synset in wn.synsets(word, pos=pos):
        for lemma in synset.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return list(synonyms)

def get_best_synset(word, sentence):
    return lesk(word_tokenize(sentence), word)

def find_contextual_synonyms(word, sentence, pos=None):
    best_synset = get_best_synset(word, sentence)
    if best_synset:
        return [lemma.name().replace('_', ' ') for lemma in best_synset.lemmas() if lemma.name() != word]
    return []

def find_punjabi_synonyms(gurmukhi_word):
    row = df[df['word'] == gurmukhi_word]
    if not row.empty:
        definition = row['definition'].values[0]
        clean_def = clean_definition(definition)
        print(f"Definition of {gurmukhi_word}: {clean_def}")
        
        split_def = split_definition(clean_def)
        
        synonyms = set()
        for phrase in split_def:
            words = word_tokenize(phrase)
            pos_tags = pos_tag(words)
            for word, pos in pos_tags:
                wordnet_pos = get_wordnet_pos(pos)
                synonyms.update(find_synonyms(word, wordnet_pos))
                synonyms.update(find_contextual_synonyms(word, phrase, wordnet_pos))
        
        possible_synonyms = []
        for synonym in synonyms:
            matching_rows = df[df['definition'].str.contains(r'\b' + re.escape(synonym) + r'\b', case=False, regex=True)]
            if not matching_rows.empty:
                possible_synonyms.extend(matching_rows['word'].values)
                #if len(possible_synonyms) >= 5:  
                #    break

        return list(set(possible_synonyms))  # Remove duplicates
    else:
        return []

# Main execution
gurmukhi_word = input("Enter a Punjabi word in Gurmukhi script: ")
synonyms = find_punjabi_synonyms(gurmukhi_word)

if synonyms:
    print(f"Possible Punjabi synonyms of {gurmukhi_word}: {synonyms}")
else:
    print(f"The word '{gurmukhi_word}' was not found in the dictionary or no synonyms were found.")