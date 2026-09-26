import pandas as pd
import re as r
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet as wn  
from nltk.corpus import stopwords
import nltk
import random

# nltk.download('wordnet')
# nltk.download('stopwords')

def map_pos(pos):
    """Maps POS tags to WordNet POS tags."""
    pos = pos.lower()
    if 'noun' in pos:
        return wn.NOUN
    elif 'verb' in pos or 'conjunct verb' in pos:
        return wn.VERB
    elif 'adjective' in pos:
        return wn.ADJ
    elif 'adverb' in pos:
        return wn.ADV
    else:
        return None

def get_synonyms(definition, pos_tag=None):
    """Retrieves synonyms from WordNet based on the definition and POS tag."""
    stop_words = set(stopwords.words('english'))
    words = r.findall(r'\b\w+\b', definition.lower())
    filtered_words = [w for w in words if w not in stop_words and len(w) > 1]
    synonyms = set()
    for word in filtered_words:
        synsets = wn.synsets(word, pos=pos_tag)
        for synset in synsets:
            for lemma in synset.lemmas():
                synonyms.add(lemma.name())
    return synonyms

def find_possible_synonyms(df, input_definition, original_pos, gurmukhi_word):
   """Finds synonyms based on TF-IDF similarity and POS matching."""
   vectorizer = TfidfVectorizer()

   # create corpus by combining all definitions with the input definition
   corpus = df['definition'].tolist() + [input_definition]
   tfidf_matrix = vectorizer.fit_transform(corpus)

   # extract TF-IDF vector for the input definition
   input_vector = tfidf_matrix[-1]

   # calculate cosine similarity between input vector and all other vectors
   similarities = cosine_similarity(input_vector, tfidf_matrix[:-1])
   most_similar_indices = similarities.argsort()[0][::-1]  # Sort by similarity (most similar first)
   possible_synonyms = []
   threshold = 0.35  # threshold for score; could be changed

   for idx in most_similar_indices:
       row = df.iloc[idx] # get the row using index
       similarity_score = similarities[0][idx] # get similarity score for this row
       if (map_pos(row['pos']) == map_pos(original_pos) and 
           row['definition'] != input_definition and 
           row['word'] != gurmukhi_word and 
           similarity_score >= threshold):  # only add if score is good enough
           possible_synonyms.append((row['word'], similarity_score))
       if len(possible_synonyms) >= 5:
           break
   return possible_synonyms

def clean_definition(text):
    """Cleans English definition of word."""
    text = r.sub(r'\([^)]*\)', '', text) # remove things in brackets
    text = r.sub(r'[^\w\s;]', '', text) # remove special characters
    text = ' '.join(text.split())
    return text.lower()

def truncate_text(text, length=40): 
    """Truncates text and adds ellipsis if needed."""
    return text[:length] + "..." if len(text) > length else text

def test_app(num_runs):
    try:
        df = pd.read_csv('punjabi_university_dictionary_test.csv')
    except FileNotFoundError:
        print("Error: 'punjabi_university_dictionary_test.csv' not found.")
        return

    for _ in range(num_runs):
        gurmukhi_word = random.choice(df['word'].tolist())  # Randomly select a Gurmukhi word

        row = df[df['word'] == gurmukhi_word]  # look up the Gurmukhi word in csv

        if not row.empty:
            original_definition = row['definition'].values[0] 
            original_pos = row['pos'].values[0]
                        
            print(f"{gurmukhi_word} ({original_pos})")  # display word, pos, and shortened definition
            print(f"DEF: {truncate_text(original_definition)}")

            cleaned_definition = clean_definition(original_definition)
            wordnet_pos = map_pos(original_pos)
            synonyms = get_synonyms(cleaned_definition, wordnet_pos)
            possible_synonyms = find_possible_synonyms(df, cleaned_definition, original_pos, gurmukhi_word) 

            if possible_synonyms:
                for syn, score in possible_synonyms:
                    syn_definition = df[df['word'] == syn]['definition'].iloc[0]
                    print(f"     [{score:.3f}] {syn} → {truncate_text(syn_definition)}")
                print("=" * 30)  # separator
            else:
                print(f"No synonyms found for '{gurmukhi_word}'.")
                print("=" * 30)
        else:
            print(f"The word '{gurmukhi_word}' was not found in the dictionary.")
            print("=" * 50)

if __name__ == "__main__":
    num_runs = int(input("Enter the number of times to run the test app: "))
    test_app(num_runs)

"""
## Script Execution Guide

### Running the Script
1. Navigate to the script directory:
   cd /Users/deep/gurmukhi/test

2. Ensure both files are present:
   - testmain.py
   - punjabi_university_dictionary_test.csv

3. Run the script:
   python3 testmain.py

### Notes
- This script is a tester for the main.py file
- Randomly selects Punjabi words from the dictionary
- Finds potential synonyms using TF-IDF and WordNet
- Displays word, part of speech, definition, and possible synonyms

### Test Parameters
- User inputs number of test runs
- Outputs synonyms with similarity scores
- Truncates long definitions for readability

### Troubleshooting
If file not found error occurs:
- Verify current working directory in the script:
  import os
  print("Current working directory:", os.getcwd())
- Use absolute path for CSV file:
  df = pd.read_csv('/Users/deep/gurmukhi/test/punjabi_university_dictionary_test.csv')

For any persistent issues, check file permissions and exact file names.
"""