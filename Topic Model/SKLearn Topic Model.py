from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis.lda_model
import pandas
from data_preprocessing import df, stop_words

# Example text data
documents = df['lemmatized_caption'].astype(str)

# Step 1: Create a document-term matrix
vectorizer = CountVectorizer(stop_words=stop_words, max_df=0.85)
doc_term_matrix = vectorizer.fit_transform(documents)

# Step 2: Train the LDA model
num_topics = 5 # Number of topics
lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda_model.fit(doc_term_matrix)

# Step 3: Interpret the topics
def display_topics(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

num_top_words = 10  # Number of top words to display for each topic
feature_names = vectorizer.get_feature_names_out()
display_topics(lda_model, feature_names, num_top_words)

import numpy as np

# Transform the documents to topic distribution
topic_distribution = lda_model.transform(doc_term_matrix)

# Assign the dominant topic to each document
df['dominant_topic'] = np.argmax(topic_distribution, axis=1)

average_likes_per_topic_LDA = df.groupby('dominant_topic')['likesCount'].mean()

print(average_likes_per_topic_LDA)

import matplotlib.pyplot as plt

average_likes_per_topic_LDA.plot(kind='bar', color='skyblue')
plt.xlabel('Topic Number')
plt.ylabel('Average Likes')
plt.title('Average Likes Per Topic')
plt.xticks(rotation=0)
plt.show()

import matplotlib.pyplot as plt

average_likes_per_topic_LDA.plot(kind='bar', color='skyblue')
plt.xlabel('Topic Number')
plt.ylabel('Average Likes')
plt.title('Average Likes Per Topic')
plt.xticks(rotation=0)
plt.show()

import pyLDAvis
import pyLDAvis.lda_model

pyLDAvis.enable_notebook()
sklearn_display = pyLDAvis.lda_model.prepare(lda_model, doc_term_matrix, vectorizer, sort_topics=False)
pyLDAvis.display(sklearn_display)