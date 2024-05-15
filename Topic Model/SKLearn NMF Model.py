from sklearn.decomposition import NMF
import ipywidgets as widgets
from IPython.display import display
from data_preprocessing import doc_term_matrix 

num_topics = 5

# Creating NMF model
nmf_model = NMF(n_components=num_topics, random_state=42)
nmf_model.fit(doc_term_matrix)

feature_names = vectorizer.get_feature_names_out()

# Display topics
def display_nmf_topics(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

def display_topic_words(n_top_words):
    display_nmf_topics(nmf_model, feature_names, n_top_words)

# Create a slider for choosing number of top words
slider = widgets.IntSlider(value=10, min=1, max=20, step=1, description='Top words:')
widgets.interactive(display_topic_words, n_top_words=slider)


import numpy as np

# Transform the documents to topic distribution
topic_distribution = nmf_model.transform(doc_term_matrix)

# Assign the dominant topic to each document
df['dominant_topic'] = np.argmax(topic_distribution, axis=1)

average_likes_per_topic = df.groupby('dominant_topic')['likesCount'].mean()

print(average_likes_per_topic)

import matplotlib.pyplot as plt

average_likes_per_topic.plot(kind='bar', color='skyblue')
plt.xlabel('Topic Number')
plt.ylabel('Average Likes')
plt.title('Average Likes Per Topic')
plt.xticks(rotation=0)
plt.show()


import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import defaultdict



def prepare_word_cloud_data(df):
    # grouping data by dominant_topic,aggregating likesCount with sum, concatenating tokens
    aggregated_data = df.groupby('dominant_topic').agg({
        'likesCount': 'sum',
        'tokenized_caption': lambda x: ' '.join([' '.join(tokens) for tokens in x])
    }).to_dict(orient='index')

    # word clouds for each topic weighted by likesCount
    for topic, data in aggregated_data.items():

        wordcloud = WordCloud(width=800, height=400, max_words=200).generate_from_text(data['tokenized_caption'])

        # display image
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Word Cloud for Topic {topic} (Total Likes: {data["likesCount"]})')
        plt.axis("off")
        plt.show()


prepare_word_cloud_data(df)


import pyLDAvis

# Ensure 'documents' is a list of all your document strings
doc_lengths = [len(doc.split()) for doc in documents]

# Vocabulary and term frequencies from the CountVectorizer
vocab = vectorizer.get_feature_names_out()
term_frequency = np.asarray(doc_term_matrix.sum(axis=0)).ravel().tolist()

# Adding a small number to avoid division by zero
epsilon = 1e-6
doc_topic_dists = nmf_model.transform(doc_term_matrix) + epsilon
doc_topic_dists = doc_topic_dists / doc_topic_dists.sum(axis=1)[:, np.newaxis]

# Normalize topic-term distributions
topic_term_dists = nmf_model.components_
topic_term_dists = topic_term_dists / topic_term_dists.sum(axis=1)[:, np.newaxis]

# Check if normalization was successful
if np.any(np.isnan(doc_topic_dists)):
    raise ValueError("NaN values found in document-topic distributions after normalization.")

# Prepare the data for pyLDAvis
data = {
    'topic_term_dists': topic_term_dists,
    'doc_topic_dists': doc_topic_dists,
    'doc_lengths': doc_lengths,
    'vocab': vocab,
    'term_frequency': term_frequency
}

# Generate the visualization
vis_data = pyLDAvis.prepare(**data)
pyLDAvis.display(vis_data)