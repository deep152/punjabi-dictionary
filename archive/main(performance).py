import pandas as pd
import re as r
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet as wn  
from nltk.corpus import stopwords
import nltk
from functools import lru_cache

class SynonymFinder:
    def __init__(self, dictionary_path):
        """Initialize the SynonymFinder with dictionary data and cached computations."""
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
        text = r.sub(r'\([^)]*\)', '', text)
        text = r.sub(r'[^\w\s;]', '', text)
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
    def get_synonyms(self, definition, pos_tag=None):
        """Retrieves synonyms from WordNet based on the definition and POS tag.
        Now cached using lru_cache for better performance."""
        words = r.findall(r'\b\w+\b', definition.lower())
        filtered_words = [w for w in words if w not in self.stop_words and len(w) > 1]
        synonyms = set()
        for word in filtered_words:
            synsets = wn.synsets(word, pos=pos_tag)
            for synset in synsets:
                for lemma in synset.lemmas():
                    synonyms.add(lemma.name())
        return tuple(synonyms)  # Convert to tuple for caching purposes

    def find_possible_synonyms(self, input_definition, original_pos, gurmukhi_word, 
                             threshold=0.35, max_synonyms=5):
        """Finds synonyms based on TF-IDF similarity and POS matching.
        Now uses pre-computed TF-IDF matrix."""
        # Transform input definition using the same vectorizer
        input_vector = self.vectorizer.transform([self.clean_definition(input_definition)])
        
        # Calculate cosine similarity using pre-computed matrix
        similarities = cosine_similarity(input_vector, self.tfidf_matrix)[0]
        
        # Create array of indices sorted by similarity
        most_similar_indices = similarities.argsort()[::-1]
        
        possible_synonyms = []
        for idx in most_similar_indices:
            row = self.df.iloc[idx]
            similarity_score = similarities[idx]
            
            if (self.map_pos(row['pos']) == self.map_pos(original_pos) and 
                row['definition'] != input_definition and 
                row['word'] != gurmukhi_word and 
                similarity_score >= threshold):
                possible_synonyms.append((row['word'], similarity_score))
            
            if len(possible_synonyms) >= max_synonyms:
                break
                
        return possible_synonyms

    @staticmethod
    def truncate_text(text, length=40):
        """Truncates text and adds ellipsis if needed."""
        return text[:length] + "..." if len(text) > length else text

    def find_word_synonyms(self, gurmukhi_word):
        """Main method to find synonyms for a given word."""
        row = self.df[self.df['word'] == gurmukhi_word]
        
        if not row.empty:
            original_definition = row['definition'].values[0]
            original_pos = row['pos'].values[0]
            
            results = {
                'word': gurmukhi_word,
                'pos': original_pos,
                'definition': self.truncate_text(original_definition),
                'synonyms': []
            }
            
            cleaned_definition = self.clean_definition(original_definition)
            wordnet_pos = self.map_pos(original_pos)
            possible_synonyms = self.find_possible_synonyms(
                cleaned_definition, original_pos, gurmukhi_word
            )
            
            if possible_synonyms:
                for syn, score in possible_synonyms:
                    syn_definition = self.df[self.df['word'] == syn]['definition'].iloc[0]
                    results['synonyms'].append({
                        'word': syn,
                        'score': score,
                        'definition': self.truncate_text(syn_definition)
                    })
            
            return results
        else:
            return None

def main():
    try:
        # Initialize the SynonymFinder with error handling
        dictionary_path = 'punjabi_university_dictionary.csv'
        print(f"Loading dictionary from {dictionary_path}...")
        finder = SynonymFinder(dictionary_path)
        print("Dictionary loaded successfully!")
        
        while True:
            try:
                gurmukhi_word = input("\nEnter a Gurmukhi word (or type 'exit'): ")
                if gurmukhi_word.lower() == 'exit':
                    break

                results = finder.find_word_synonyms(gurmukhi_word)
                
                if results:
                    print(f"{results['word']} ({results['pos']})")
                    print(f"DEF: {results['definition']}")
                    
                    if results['synonyms']:
                        for syn in results['synonyms']:
                            print(f"     [{syn['score']:.3f}] {syn['word']} → {syn['definition']}")
                        print("=" * 30)
                    else:
                        print(f"No synonyms found for '{gurmukhi_word}'.")
                        print("=" * 30)
                else:
                    print(f"The word '{gurmukhi_word}' was not found in the dictionary.")
                    print("=" * 50)
                    
            except KeyboardInterrupt:
                print("\nProgram interrupted by user.")
                break
            except Exception as e:
                print(f"An error occurred while processing the word: {str(e)}")
                print("Please try again with a different word.")
                
    except FileNotFoundError:
        print(f"Error: The dictionary file '{dictionary_path}' was not found.")
        print("Please ensure the file exists in the correct location.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        print("Please check your setup and try again.")
    finally:
        print("\nThank you for using the Gurmukhi Synonym Finder!")

if __name__ == "__main__":
    main()