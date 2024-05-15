import os
import string
import pandas
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from cleantext import clean
#nltk.download('all')


# reads json file containing caption
project_folder = os.getcwd()
file_name = input("Please enter the file you would like to process and model. ")
file_path = os.path.join(project_folder, 'data', file_name)

if os.path.exists(file_path):
    df = pandas.read_json(file_path)
else: 
    print("File not found.")    


custom_stop_words = ['startupfiu', 'fiu', 'startup', 'http', 'skydeck', 'berkeley', 'miami', 'ca', 'us','cal', 'uc']
std_stop_words = stopwords.words('english')
std_stop_words += list(string.punctuation)
stop_words = custom_stop_words + std_stop_words

#tokenizes, removes: emoji, numbers, punctuation, and makes all words lowercase
def tokenize_lowercase(text):
    tokens = word_tokenize(text)
    new_text = [word for word in tokens if word.isalpha()]
    stopwords_removed = [token.lower() for token in new_text if token.lower() not in stop_words]
    return stopwords_removed

df['caption'] = df['caption'].apply(tokenize_lowercase)

def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

lemmatizer = WordNetLemmatizer()

def lemmatize_sentence(sentence):
    if isinstance(sentence, list):  # If input is a list of tokens
        sentence = " ".join(sentence)  # Convert list to string
    # Tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    # Convert nltk POS tags to WordNet tags
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    # Lemmatize each word based on its POS tag
    lemmatized_tokens = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_tokens.append(word)
        else:
            lemmatized_tokens.append(lemmatizer.lemmatize(word, tag))
    return lemmatized_tokens

# Apply the lemmatization function directly to the 'caption' column of the DataFrame
df['lemmatized_caption'] = df['caption'].apply(lemmatize_sentence)