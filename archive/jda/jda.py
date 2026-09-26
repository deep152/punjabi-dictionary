import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the dictionary
try:
    df = pd.read_csv('punjabi_university_dictionary.csv')
    logger.info(f"Dictionary loaded successfully. Total entries: {len(df)}")
except Exception as e:
    logger.error(f"Error loading dictionary: {e}")
    raise

def map_pos(pos):
    """
    Map Punjabi POS to WordNet categories with extended support.
    
    Args:
    pos (str): Part of speech in Punjabi

    Returns:
    str: Mapped WordNet POS or None
    """
    pos = pos.lower()
    pos_mappings = {
        'noun': wn.NOUN,
        'verb': wn.VERB,
        'adjective': wn.ADJ,
        'adverb': wn.ADV,
        'conjunct verb': wn.VERB,
        'phrase': 'PHRASE',
        'compound': 'COMPOUND',
        'pronoun': wn.NOUN,
        'preposition': 'PREP',
        'interjection': 'INTERJ',
        'conjunction': 'CONJ'
    }
    return pos_mappings.get(pos, None)

def smart_truncate(text, length=40):
    """
    Intelligently truncate text considering word boundaries.
    
    Args:
    text (str): Text to truncate
    length (int): Maximum length of truncated text

    Returns:
    str: Truncated text
    """
    if len(text) <= length:
        return text
    truncated = text[:length].rsplit(' ', 1)[0]
    return truncated + "..."

def preprocess_text(text):
    """
    Preprocess text for TF-IDF vectorization.
    
    Args:
    text (str): Input text

    Returns:
    str: Preprocessed text
    """
    text = re.sub(r'[^\w\s]', '', text.lower())
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words])

def find_possible_synonyms(df, input_definition, original_pos, gurmukhi_word):
    """
    Find possible synonyms using TF-IDF and cosine similarity.
    
    Args:
    df (DataFrame): Dictionary DataFrame
    input_definition (str): Definition of the input word
    original_pos (str): POS of the input word
    gurmukhi_word (str): Input Gurmukhi word

    Returns:
    list: List of tuples containing synonyms, scores, and definitions
    """
    vectorizer = TfidfVectorizer(strip_accents='unicode', lowercase=True, stop_words='english')
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
            possible_synonyms.append((row['word'], similarity_score, row['definition']))
        if len(possible_synonyms) >= 5:
            break
    return possible_synonyms

def handle_reference_chains(word, definition):
    """
    Handle 'same as' references in definitions.
    
    Args:
    word (str): Input word
    definition (str): Definition of the word

    Returns:
    tuple: Updated word and definition
    """
    if "same as" in definition.lower():
        referenced_word = definition.split("same as")[-1].strip()
        return find_word(referenced_word)
    return word, definition

def find_word(word):
    """
    Find a word in the dictionary.
    
    Args:
    word (str): Word to find

    Returns:
    tuple: Word and its definition, or (None, None) if not found
    """
    row = df[df['word'] == word]
    if not row.empty:
        return row.iloc[0]['word'], row.iloc[0]['definition']
    return None, None

def analyze_score_distribution(synonyms):
    """
    Analyze the distribution of similarity scores.
    
    Args:
    synonyms (list): List of synonym tuples

    Returns:
    dict: Score distribution in different ranges
    """
    score_ranges = {
        'perfect': 0,  # 1.000
        'high': 0,     # 0.7-0.999
        'medium': 0,   # 0.5-0.699
        'threshold': 0 # 0.35-0.499
    }
    
    for _, score, _ in synonyms:
        if score == 1.0:
            score_ranges['perfect'] += 1
        elif score >= 0.7:
            score_ranges['high'] += 1
        elif score >= 0.5:
            score_ranges['medium'] += 1
        else:
            score_ranges['threshold'] += 1
            
    return score_ranges

def identify_patterns(synonyms):
    """
    Identify patterns in synonym results.
    
    Args:
    synonyms (list): List of synonym tuples

    Returns:
    dict: Identified patterns
    """
    patterns = {
        'quality_decay': False,
        'domain_consistent': True,
        'pos_matched': True
    }
    
    scores = [score for _, score, _ in synonyms]
    if all(scores[i] >= scores[i+1] for i in range(len(scores)-1)):
        patterns['quality_decay'] = True
        
    return patterns

def handle_calendar_terms(word, definition):
    """
    Special handling for calendar and time-related terms.
    
    Args:
    word (str): Input word
    definition (str): Definition of the word

    Returns:
    list: List of related calendar terms
    """
    calendar_terms = ['month', 'day', 'festival', 'season']
    if any(term in definition.lower() for term in calendar_terms):
        # Implement special logic for calendar terms
        pass
    return []

def handle_technical_terms(word, definition):
    """
    Special handling for technical and domain-specific terms.
    
    Args:
    word (str): Input word
    definition (str): Definition of the word

    Returns:
    list: List of related technical terms
    """
    technical_domains = ['scientific', 'mathematical', 'medical', 'legal']
    if any(domain in definition.lower() for domain in technical_domains):
        # Implement special logic for technical terms
        pass
    return []

def punjabi_synonym_finder(gurmukhi_word):
    """
    Main function to find synonyms for a given Punjabi word.
    
    Args:
    gurmukhi_word (str): Input Gurmukhi word

    Returns:
    None (prints results)
    """
    row = df[df['word'] == gurmukhi_word]
    if row.empty:
        logger.warning(f"Word '{gurmukhi_word}' not found in the dictionary.")
        return

    word = row.iloc[0]['word']
    pos = row.iloc[0]['pos']
    definition = row.iloc[0]['definition']

    word, definition = handle_reference_chains(word, definition)
    
    possible_synonyms = find_possible_synonyms(df, definition, pos, word)
    
    print(f"{word} ({pos})")
    print(f"DEF: {smart_truncate(definition)}")
    
    for syn, score, syn_def in possible_synonyms:
        print(f"     [{score:.3f}] {syn} → {smart_truncate(syn_def)}")
    
    print("=" * 30)
    
    # Quality metrics
    score_distribution = analyze_score_distribution(possible_synonyms)
    patterns = identify_patterns(possible_synonyms)
    
    print("Quality Metrics:")
    print(f"Score Distribution: {score_distribution}")
    print(f"Patterns: {patterns}")

    # Special handling for calendar and technical terms
    calendar_related = handle_calendar_terms(word, definition)
    technical_related = handle_technical_terms(word, definition)

    if calendar_related:
        print("Related Calendar Terms:")
        for term in calendar_related:
            print(f"  - {term}")

    if technical_related:
        print("Related Technical Terms:")
        for term in technical_related:
            print(f"  - {term}")

def batch_process(word_list):
    """
    Process a batch of words for synonym finding.
    
    Args:
    word_list (list): List of Gurmukhi words to process

    Returns:
    None (processes each word)
    """
    for word in word_list:
        punjabi_synonym_finder(word)
        print("\n")

def performance_analysis():
    """
    Analyze and report on the performance of the synonym finder.
    
    Returns:
    dict: Performance metrics
    """
    # Implement performance analysis logic
    pass

def export_results(results, filename):
    """
    Export synonym finding results to a file.
    
    Args:
    results (dict): Results to export
    filename (str): Name of the file to export to

    Returns:
    None (exports to file)
    """
    # Implement export logic
    pass

if __name__ == "__main__":
    # Example usage
    test_words = ["ਉਪਵਾਸ", "ਹਸਤਖੇਪ", "ਦਿਹਾਂਤ", "ਜਿੱਕਣ", "ਛਤੀਰ"]
    batch_process(test_words)

    # Performance analysis
    perf_metrics = performance_analysis()
    print("Performance Metrics:")
    print(perf_metrics)

    # Export results
    export_results(perf_metrics, "synonym_finder_results.json")