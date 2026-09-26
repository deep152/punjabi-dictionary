import pandas as pd
import re as r
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet as wn  
from nltk.corpus import stopwords
import nltk


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
    """Truncates text and adds ellipsis if needed.
    This is kind of unneccaasry but its just for when I was running a lot of words at the same time it was getting hard to read
    """
    return text[:length] + "..." if len(text) > length else text

def main():
    try:
        df = pd.read_csv('punjabi_university_dictionary.csv')
    except FileNotFoundError:
        print("Error: 'punjabi_university_dictionary.csv' not found.")
        return

    while True:
        gurmukhi_word = input("\nEnter a Gurmukhi word (or type 'exit'): ")
        if gurmukhi_word.lower() == 'exit':
            break

        row = df[df['word'] == gurmukhi_word] # look up the Gurmukhi word in csv

        if not row.empty:
            original_definition = row['definition'].values[0] 
            original_pos = row['pos'].values[0]
                        
            print(f"{gurmukhi_word} ({original_pos})") # display word, pos, and shortened definition
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
    main()

"""
handling for phrases
words with calendar terms (sometimes mixes different temporal units) 
Months matching with days of week; Festival dates (holidays) mixing with regular calendar dates

some of these things aren't that big of a deal but I just wrote them down
"""