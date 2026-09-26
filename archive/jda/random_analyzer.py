import pandas as pd
import re as r
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet as wn  
from nltk.corpus import stopwords
import nltk
import random
import time

# Your existing functions
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
    corpus = df['definition'].tolist() + [input_definition]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    input_vector = tfidf_matrix[-1]
    similarities = cosine_similarity(input_vector, tfidf_matrix[:-1])
    most_similar_indices = similarities.argsort()[0][::-1]
    possible_synonyms = []
    threshold = 0.35

    for idx in most_similar_indices:
        row = df.iloc[idx]
        similarity_score = similarities[0][idx]
        if (map_pos(row['pos']) == map_pos(original_pos) and 
            row['definition'] != input_definition and 
            row['word'] != gurmukhi_word and 
            similarity_score >= threshold):
            possible_synonyms.append((row['word'], similarity_score))
        if len(possible_synonyms) >= 5:
            break
    return possible_synonyms

def clean_definition(text):
    """Cleans English definition of word."""
    text = r.sub(r'\([^)]*\)', '', text)
    text = r.sub(r'[^\w\s;]', '', text)
    text = ' '.join(text.split())
    return text.lower()

def truncate_text(text, length=40):
    """Truncates text and adds ellipsis if needed."""
    return text[:length] + "..." if len(text) > length else text

def analyze_random_words(num_words=20, delay=1):
    """Analyzes a specified number of random words from the dictionary."""
    try:
        # Read the dictionary
        df = pd.read_csv('punjabi_university_dictionary.csv')
        
        # Get random unique indices
        random_indices = random.sample(range(len(df)), num_words)
        
        print(f"\nAnalyzing {num_words} random words...")
        print("Press Ctrl+C to stop at any time.\n")
        
        for i, idx in enumerate(random_indices, 1):
            row = df.iloc[idx]
            gurmukhi_word = row['word']
            
            print(f"Enter a Gurmukhi word (or type 'exit'): {gurmukhi_word}")
            
            original_definition = row['definition']
            original_pos = row['pos']
            
            # Format POS for display
            pos_display = original_pos.lower()
            
            print(f"{gurmukhi_word} ({pos_display})")
            print(f"DEF: {truncate_text(original_definition)}")

            cleaned_definition = clean_definition(original_definition)
            wordnet_pos = map_pos(original_pos)
            possible_synonyms = find_possible_synonyms(df, cleaned_definition, original_pos, gurmukhi_word)

            if possible_synonyms:
                for syn, score in possible_synonyms:
                    syn_definition = df[df['word'] == syn]['definition'].iloc[0]
                    print(f"     [{score:.3f}] {syn} → {truncate_text(syn_definition)}")
                print("=" * 30)
            else:
                print(f"No synonyms found for '{gurmukhi_word}'.")
                print("=" * 30)
                
            # Add a delay between words
            time.sleep(delay)

    except FileNotFoundError:
        print("Error: 'punjabi_university_dictionary.csv' not found.")
    except KeyboardInterrupt:
        print("\nAnalysis stopped by user.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    analyze_random_words(num_words=20, delay=1)