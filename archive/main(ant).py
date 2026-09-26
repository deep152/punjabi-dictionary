import pandas as pd
import re as r
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet as wn  
from nltk.corpus import stopwords
import nltk
from functools import lru_cache

class AntonymFinder:
    def __init__(self, dictionary_path):
        """Initialize the AntonymFinder with dictionary data and cached computations."""
        try:
            self.df = pd.read_csv(dictionary_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dictionary file not found: {dictionary_path}")
        except pd.errors.EmptyDataError:
            raise ValueError("The dictionary file is empty")
        except pd.errors.ParserError:
            raise ValueError("Error parsing the dictionary file. Please check the CSV format")
        
        self.vectorizer = TfidfVectorizer()
        
        # Pre-clean all definitions
        self.df['cleaned_definition'] = self.df['definition'].apply(self.clean_definition)
        
        # Pre-compute TF-IDF matrix for all definitions
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['cleaned_definition'])
        
        # Cache for storing computed similarities
        self.similarity_cache = {}
        
        # Initialize stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            print("Downloading NLTK stopwords...")
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))

    @staticmethod
    def clean_definition(text):
        """Cleans English definition of word."""
        text = r.sub(r'\([^)]*\)', '', text)  # remove things in brackets
        text = r.sub(r'[^\w\s;]', '', text)   # remove special characters
        text = ' '.join(text.split())
        return text.lower()

    @staticmethod
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

    @lru_cache(maxsize=1000)
    def get_antonyms(self, word, pos_tag=None):
        """Get antonyms for a word using WordNet."""
        antonyms = set()
        for synset in wn.synsets(word, pos=pos_tag):
            for lemma in synset.lemmas():
                if lemma.antonyms():
                    antonyms.update(ant.name() for ant in lemma.antonyms())
        return tuple(antonyms)

    def find_possible_antonyms(self, input_definition, original_pos, gurmukhi_word, 
                             threshold=0.35, max_results=5):
        """Finds potential antonyms based on semantic opposition."""
        # Transform input definition using the same vectorizer
        input_vector = self.vectorizer.transform([self.clean_definition(input_definition)])
        
        # Calculate cosine similarity using pre-computed matrix
        similarities = cosine_similarity(input_vector, self.tfidf_matrix)[0]
        
        # Create array of indices sorted by similarity (lower similarity might indicate opposition)
        most_different_indices = similarities.argsort()  # Sort ascending for antonyms
        
        possible_antonyms = []
        for idx in most_different_indices:
            row = self.df.iloc[idx]
            opposition_score = 1 - similarities[idx]  # Convert similarity to opposition
            
            if (self.map_pos(row['pos']) == self.map_pos(original_pos) and 
                row['definition'] != input_definition and 
                row['word'] != gurmukhi_word and 
                opposition_score >= threshold):
                
                possible_antonyms.append((row['word'], opposition_score))
            
            if len(possible_antonyms) >= max_results:
                break
                
        return possible_antonyms

    @staticmethod
    def truncate_text(text, length=40):
        """Truncates text and adds ellipsis if needed."""
        return text[:length] + "..." if len(text) > length else text

    def find_word_antonyms(self, gurmukhi_word):
        """Main method to find antonyms for a given word."""
        row = self.df[self.df['word'] == gurmukhi_word]
        
        if not row.empty:
            original_definition = row['definition'].values[0]
            original_pos = row['pos'].values[0]
            
            results = {
                'word': gurmukhi_word,
                'pos': original_pos,
                'definition': self.truncate_text(original_definition),
                'antonyms': []
            }
            
            cleaned_definition = self.clean_definition(original_definition)
            wordnet_pos = self.map_pos(original_pos)
            
            # Get key words from definition
            key_words = [word for word in cleaned_definition.split() 
                        if word not in self.stop_words and len(word) > 2]
            
            # Get direct antonyms from WordNet for key terms
            wordnet_antonyms = set()
            for word in key_words:
                antonyms = self.get_antonyms(word, wordnet_pos)
                if antonyms:
                    wordnet_antonyms.update(antonyms)
            
            # Find potential antonyms based on semantic opposition
            possible_antonyms = self.find_possible_antonyms(
                cleaned_definition, original_pos, gurmukhi_word
            )
            
            if possible_antonyms:
                for word, score in possible_antonyms:
                    word_def = self.df[self.df['word'] == word]['definition'].iloc[0]
                    results['antonyms'].append({
                        'word': word,
                        'score': score,
                        'definition': self.truncate_text(word_def),
                        'wordnet_antonyms': [ant for ant in wordnet_antonyms if ant in word_def.lower()]
                    })
            
            return results
        else:
            return None

def main():
    try:
        print("Initializing Antonym Finder...")
        finder = AntonymFinder('punjabi_university_dictionary.csv')
        print("Initialization complete!")
        
        while True:
            gurmukhi_word = input("\nEnter a Gurmukhi word (or type 'exit'): ")
            if gurmukhi_word.lower() == 'exit':
                break

            results = finder.find_word_antonyms(gurmukhi_word)
            
            if results:
                print(f"\n{results['word']} ({results['pos']})")
                print(f"DEF: {results['definition']}")
                
                if results['antonyms']:
                    for ant in results['antonyms']:
                        print(f"     [{ant['score']:.3f}] {ant['word']} ⟷ {ant['definition']}")
                        if ant['wordnet_antonyms']:
                            print(f"        Direct antonyms found: {', '.join(ant['wordnet_antonyms'])}")
                    print("=" * 30)
                else:
                    print(f"No antonyms found for '{gurmukhi_word}'.")
                    print("=" * 30)
            else:
                print(f"The word '{gurmukhi_word}' was not found in the dictionary.")
                print("=" * 50)
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
if __name__ == "__main__":
    main()