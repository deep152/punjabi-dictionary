import pandas as pd
import re as r
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet as wn  
from nltk.corpus import stopwords
import nltk
import logging
from typing import List, Tuple, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SynonymFinder:
    def __init__(self, csv_path: str):
        """Initialize with configuration and quality settings"""
        self.threshold = 0.35
        self.max_synonyms = 5
        self.truncate_length = 40
        
        try:
            self.df = pd.read_csv(csv_path)
            logger.info(f"Dictionary loaded successfully. Total entries: {len(self.df)}")
        except Exception as e:
            logger.error(f"Error loading dictionary: {e}")
            raise

    def map_pos(self, pos: str) -> Optional[str]:
        """Enhanced POS mapping with extended support"""
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

    def get_synonyms(self, definition: str, pos_tag: Optional[str] = None) -> set:
        """Get synonyms from WordNet with better filtering"""
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

    def find_possible_synonyms(self, 
                             input_definition: str, 
                             original_pos: str,
                             gurmukhi_word: str) -> List[Tuple[str, float]]:
        """Enhanced synonym finding with quality control"""
        vectorizer = TfidfVectorizer()
        corpus = self.df['definition'].tolist() + [input_definition]
        tfidf_matrix = vectorizer.fit_transform(corpus)
        input_vector = tfidf_matrix[-1]
        similarities = cosine_similarity(input_vector, tfidf_matrix[:-1])
        most_similar_indices = similarities.argsort()[0][::-1]
        possible_synonyms = []

        for idx in most_similar_indices:
            row = self.df.iloc[idx]
            similarity_score = similarities[0][idx]
            if (self.map_pos(row['pos']) == self.map_pos(original_pos) and 
                row['definition'] != input_definition and 
                row['word'] != gurmukhi_word and 
                similarity_score >= self.threshold):
                possible_synonyms.append((row['word'], similarity_score))
            if len(possible_synonyms) >= self.max_synonyms:
                break

        return possible_synonyms

    def truncate_text(self, text: str) -> str:
        """Smart text truncation"""
        if len(text) <= self.truncate_length:
            return text
        return text[:self.truncate_length] + "..."

    def analyze_quality(self, synonyms: List[Tuple[str, float]]) -> Dict:
        """Analyze synonym quality and patterns"""
        quality_metrics = {
            'score_distribution': {
                'high': len([s for s, score in synonyms if score >= 0.7]),
                'medium': len([s for s, score in synonyms if 0.5 <= score < 0.7]),
                'low': len([s for s, score in synonyms if score < 0.5])
            },
            'average_score': sum(score for _, score in synonyms) / len(synonyms) if synonyms else 0,
            'quality_decay': all(synonyms[i][1] >= synonyms[i+1][1] 
                               for i in range(len(synonyms)-1)) if len(synonyms) > 1 else False
        }
        return quality_metrics

    def process_word(self, gurmukhi_word: str) -> None:
        """Main processing function with enhanced output"""
        row = self.df[self.df['word'] == gurmukhi_word]

        if not row.empty:
            original_definition = row['definition'].values[0]
            original_pos = row['pos'].values[0]
            
            print(f"{gurmukhi_word} ({original_pos.lower()})")
            print(f"DEF: {self.truncate_text(original_definition)}")

            cleaned_definition = self.clean_definition(original_definition)
            wordnet_pos = self.map_pos(original_pos)
            synonyms = self.find_possible_synonyms(cleaned_definition, original_pos, gurmukhi_word)

            if synonyms:
                for syn, score in synonyms:
                    syn_definition = self.df[self.df['word'] == syn]['definition'].iloc[0]
                    print(f"     [{score:.3f}] {syn} → {self.truncate_text(syn_definition)}")
                
                # Quality analysis
                quality_metrics = self.analyze_quality(synonyms)
                logger.info(f"Quality metrics for {gurmukhi_word}: {quality_metrics}")
            else:
                print(f"No synonyms found for '{gurmukhi_word}'.")
            
            print("=" * 30)
        else:
            print(f"The word '{gurmukhi_word}' was not found in the dictionary.")
            print("=" * 30)

    def clean_definition(self, text: str) -> str:
        """Enhanced definition cleaning"""
        text = r.sub(r'\([^)]*\)', '', text)
        text = r.sub(r'[^\w\s;]', '', text)
        return ' '.join(text.split()).lower()

def main():
    try:
        finder = SynonymFinder('punjabi_university_dictionary.csv')
        
        while True:
            gurmukhi_word = input("\nEnter a Gurmukhi word (or type 'exit'): ")
            if gurmukhi_word.lower() == 'exit':
                break
                
            finder.process_word(gurmukhi_word)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()