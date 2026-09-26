import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import nltk


#Maps parts of speech tags to wordnet part of speech tags.

def map_pos(pos):
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

#Retrieves synonyms from WordNet based on definition and part of speech tag.

def get_synonyms(definition, pos_tag=None):
    stop_words = set(stopwords.words('english'))
    words = re.findall(r'\b\w+\b', definition.lower())
    filtered_words = [w for w in words if w not in stop_words and len(w) > 1]
    synonyms = set()
    for word in filtered_words:
        synsets = wn.synsets(word, pos=pos_tag)
        for synset in synsets:
            for lemma in synset.lemmas():
                synonyms.add(lemma.name())
    return synonyms

# Define reference patterns with named capture groups for clarity and flexibility
REF_PATTERNS = {
    "simple": r"(?:same as|see|cf\.|also) (?P<ref_word>\S+)",
    "parenthetical": r"\((?:same as|see) (?P<ref_word>\S+)\)",
    "double":r"(?:same as) (?P<ref_word1>\S+) or (?P<ref_word2>\S+)" # captures multiple references
}


def resolve_reference_definition(df, word, definition, depth=0, max_depth=3): 
    """Handles definition references recursively, up to a specified depth."""

    if depth > max_depth:  # Base case to prevent infinite recursion
         return definition  
    

    for pattern_type, pattern in REF_PATTERNS.items():

        match = re.search(pattern, definition, re.IGNORECASE)


        if match:
            if pattern_type == 'double':
                #Handle both references
                 resolved_def1 = resolve_reference(df, match.group('ref_word1'), depth + 1)  # Recursive
                 resolved_def2 = resolve_reference(df, match.group('ref_word2'), depth + 1)

                 #Combines defintions, handling cases where one or both references are not found.

                 if resolved_def1 and resolved_def2:
                       return f"{resolved_def1}; {resolved_def2}"# Or other suitable combiner.

                 elif resolved_def1:
                       return resolved_def1
                 elif resolved_def2:
                     return resolved_def2
                 else:  # Neither ref found
                    return definition




            else: 

                resolved_def = resolve_reference(df, match.group('ref_word'), depth + 1)  # Recursive call
                if resolved_def: 
                   return resolved_def

                # Handles circular references; If a reference points back to the original word.

                elif resolved_def == definition: 
                       return definition

    return definition  # No more references found



def resolve_reference(df, ref_word, depth):

      referenced_row = df[df['word'] == ref_word]
      if not referenced_row.empty:
           return referenced_row['definition'].iloc[0]
      return None   #Explicitly return None if the reference word is not found


def find_possible_synonyms(df, input_word, input_definition, original_pos, max_synonyms=3):

    vectorizer = TfidfVectorizer()

    #Filter out rows where definitions are empty/very short after cleaning or contain only numbers.
    
    df_filtered = df[df['definition'].apply(lambda x: len(clean_definition(x)) > 2 and not clean_definition(x).isdigit())] # filters empty string or string of ints


    corpus = df_filtered['definition'].tolist() + [input_definition]

    try: 
       tfidf_matrix = vectorizer.fit_transform(corpus)

    except ValueError: 
        print(f"ValueError: After cleaning and removing empty, all defintions are numbers for {input_word}. Skipping.")  # Prints diagnostic message
        return []    #Or handle appropriately.

    input_vector = tfidf_matrix[-1]


    similarities = []
    seen_definitions = set()
    
    for idx, row in df_filtered.iterrows():
        if row['word'] == input_word: 
           continue


        actual_def = resolve_reference_definition(df, row['word'], row['definition'])

        similarity = cosine_similarity(vectorizer.transform([actual_def]), input_vector)[0][0]

        if map_pos(row['pos']) == map_pos(original_pos):
            similarity *= 1.2   # Boost synonyms from matching POS   

        if similarity > 0.25:  #Adjusted for better precesion
             cleaned_sim_def = clean_definition(actual_def) # Cleaning for consistent comparisons
             if cleaned_sim_def not in seen_definitions: 
                   similarities.append((row['word'], similarity))  # Store the actual word
                   seen_definitions.add(cleaned_sim_def)       



    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:max_synonyms]  



def clean_definition(text):


    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'[^\w\s;।॥]', '', str(text)) # Expanded punctuation removal
    text = ' '.join(text.split())
    return text.lower()





#The main function of the program
def main():
   nltk.download('omw-1.4') # Ensures that wordnet data has been downloaded. This only needs to run once
   try:
        df = pd.read_csv('punjabi_university_dictionary.csv')
   except FileNotFoundError:
        print("Error: 'punjabi_university_dictionary.csv' not found.")
        return

   while True:
        gurmukhi_word = input("Enter a Gurmukhi word (or type 'exit'): ")
        if gurmukhi_word.lower() == 'exit':
            break


        rows = df[df['word'] == gurmukhi_word].copy() # Create a copy

        if not rows.empty:

            #Iterate through rows if input word has multiple POS and defenitions

            for index, row in rows.iterrows(): # iterates if same word has different defintions with different POS 
                original_definition = row['definition']
                original_pos = row['pos']
                # Resolve the input word's definition before cleaning or synonym lookup
                resolved_definition = resolve_reference_definition(df, gurmukhi_word, original_definition)

                cleaned_definition = clean_definition(resolved_definition)


                wordnet_pos = map_pos(original_pos)
                synonyms = get_synonyms(cleaned_definition, wordnet_pos)

                possible_synonyms = find_possible_synonyms(df, gurmukhi_word, cleaned_definition, original_pos)


                if possible_synonyms:
                      print(f"Definition of {gurmukhi_word} ({original_pos}): {resolved_definition}")
                      print(f"\nSynonyms of {gurmukhi_word} ({original_pos}):")
                      for i, (synonym, score) in enumerate(possible_synonyms, 1):

                          synonym_definition = df[df['word'] == synonym]['definition'].iloc[0]


                          resolved_syn_def = resolve_reference_definition(df, synonym, synonym_definition)


                          print(f"Syn {i}: {synonym} ({df[df['word']==synonym]['pos'].iloc[0]}) - {resolved_syn_def} (Similarity: {score:.2f})")  # Display syn POS and score

                else:

                    print(f"Definition of {gurmukhi_word} ({original_pos}): {resolved_definition}")
                    print(f"\nNo sufficiently similar synonyms found for '{gurmukhi_word}' ({original_pos}).")



        else:
            print(f"The word '{gurmukhi_word}' was not found in the dictionary.")

if __name__ == "__main__":
    main()