import pandas as pd
import re

def clean_sentences(sentences):
   cleaned_sentences = []
   for sentence in sentences:
       cleaned = ' '.join(sentence.split())
       cleaned_sentences.append(cleaned)
   return cleaned_sentences

def get_word_commonness_score(word, common_words):
   """get commonness of sentence according to common words list"""
   if word not in common_words:
       return 0.1
   position = common_words.index(word)
   if position < 20:
       return 1.0
   elif position < 50:
       return 0.9
   elif position < 100:
       return 0.8
   elif position < 220:
       return 0.7
   return 0.6

def score_sentence(sentence, target_word, common_words):
   """Returns commonness and length scores separately"""
   words = sentence.split()
   
   # Penalize repetition of target word
   if words.count(target_word) > 1:
       return (0.1, 0.1)  # Very low scores for both if repeated
       
   # Calculate commonness score
   scoring_words = [w for w in words if w != target_word]
   word_scores = [get_word_commonness_score(word, common_words) for word in scoring_words]
   commonness_score = sum(word_scores) / len(word_scores) if word_scores else 0
   
   # Calculate length score
   length = len(words)
   if 7 <= length <= 12:
       length_score = 1.0  # Best length
   elif 4 <= length < 7:
       length_score = 0.6  # OK but short
   elif length < 4:
       length_score = 0.3  # Too short
   else:
       length_score = max(0.1, 1.0 - (length - 12) * 0.05)
       
   return (commonness_score, length_score)

def get_word_sentence_examples(word, sentences, max_examples, common_words):
    scored_sentences = []
    cleaned_sentences = clean_sentences(sentences)
    
    for original, cleaned in zip(sentences, cleaned_sentences):
        # Skip sentences with problematic punctuation
        if "..." in original or "…" in original:
            continue
            
        if word in cleaned.split():
            commonness_score, length_score = score_sentence(cleaned, word, common_words)
            # Could adjust these thresholds
            if length_score >= 0.3:  # Allow shorter sentences but show score
                final_score = (commonness_score * 0.6) + (length_score * 0.4)
                scored_sentences.append((original, final_score, commonness_score, length_score))
                
    return sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:max_examples]

def main():
   common_words = pd.read_csv('500_common_punjabi_words.csv')['Word'].tolist()[1:]
   sentences_df = pd.read_csv('punjabi_sentences (1).csv')
   sentences = sentences_df.iloc[:, 1].tolist()
   
   while True:
       word = input("\nEnter a Punjabi word to find examples (or 'q' to quit): ")
       if word.lower() == 'exit':
           break
           
       results = get_word_sentence_examples(word, sentences, 5, common_words)
       if results:
           print(f"\nExamples for '{word}':")
           for sentence, final_score, commonness_score, length_score in results:
               print(f"[Final: {final_score:.3f}, Common: {commonness_score:.3f}, Length: {length_score:.3f}] {sentence}")
       else:
           print(f"\nNo examples found for '{word}'")

if __name__ == "__main__":
   main()