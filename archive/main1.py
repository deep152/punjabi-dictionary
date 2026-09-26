import csv
import re

def load_common_words(filename):
    """Load the 500 most common Punjabi words"""
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()[1:]]  # Skip header 'Word'

def load_sentences(filename):
    """Load sentences from CSV file"""
    sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if row and len(row) > 1:
                sentences.append(row[1])
    return sentences

def preprocess_sentences(sentences):
    """Basic cleaning of sentences - handling whitespace"""
    cleaned_sentences = []
    for sentence in sentences:
        cleaned = ' '.join(sentence.split())
        cleaned_sentences.append(cleaned)
    return cleaned_sentences

def get_word_commonness_score(word, common_words):
    """Score word based on its commonness with debug output"""
    try:
        position = common_words.index(word)
        if position < 20:
            score = 1.0
        elif position < 50:
            score = 0.8
        elif position < 100:
            score = 0.6
        elif position < 220:
            score = 0.4
        elif position < 500:
            score = 0.2
        else:
            score = 0.1
        print(f"Word: {word}, Position: {position}, Score: {score}")
        return score
    except ValueError:
        print(f"Word: {word} not in common words, Score: 0.1")
        return 0.1

def score_sentence(sentence, target_word, common_words):
    """Score a sentence based on how common its words are and length"""
    words = sentence.split()
    print(f"\nScoring sentence: {sentence}")
    print(f"Target word: {target_word}")
    
    # Don't count target_word in scoring
    scoring_words = [w for w in words if w != target_word]
    print(f"Words being scored: {scoring_words}")
    
    # Word commonness score (60% of total score)
    word_scores = [get_word_commonness_score(word, common_words) for word in scoring_words]
    commonness_score = sum(word_scores) / len(word_scores) if word_scores else 0
    print(f"Average commonness score: {commonness_score}")
    
    # Length score (40% of total score)
    length = len(words)
    if 4 <= length <= 12:
        length_score = 1.0
    elif length < 4:
        length_score = 0.3
    else:
        length_score = max(0.1, 1.0 - (length - 12) * 0.05)
    print(f"Length score: {length_score}")
    
    final_score = (commonness_score * 0.6) + (length_score * 0.4)
    print(f"Final score: {final_score}\n")
    
    return final_score

def get_word_sentence_examples(word, sentences, max_examples, common_words):
    """Get best example sentences for a word"""
    scored_sentences = []
    cleaned_sentences = preprocess_sentences(sentences)
    
    for original, cleaned in zip(sentences, cleaned_sentences):
        if word in cleaned.split():
            score = score_sentence(cleaned, word, common_words)
            scored_sentences.append((original, score))
    
    return sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:max_examples]

def main():
    try:
        common_words = load_common_words('500_common_punjabi_words.csv')
        sentences = load_sentences('punjabi_sentences (1).csv')
        
        while True:
            word = input("\nEnter a Punjabi word to find examples (or 'exit' to quit): ")
            if word.lower() == 'exit':
                break
                
            results = get_word_sentence_examples(word, sentences, 5, common_words)
            
            if results:
                print(f"\nTop example sentences for '{word}':")
                for i, (sentence, score) in enumerate(results, 1):
                    print(f"\n{i}. Score: {score:.3f}")
                    print(f"   {sentence}")
            else:
                print(f"\nNo examples found for '{word}'")
                
    except FileNotFoundError as e:
        print(f"Error: Could not find required file - {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()