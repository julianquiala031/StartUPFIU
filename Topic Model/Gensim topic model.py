from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import pandas
from data_preprocessing import df

# Read data from JSON file into DataFrame
df['tokenized_caption'] = df['lemmatized_caption']

# Create a dictionary from the preprocessed captions
dictionary = Dictionary(df['tokenized_caption'])

# Filter out tokens that appear in less than 5 documents or more than 50% of documents
dictionary.filter_extremes(no_below=5, no_above=0.5)

# Convert the captions to bag-of-words representation
corpus = [dictionary.doc2bow(caption) for caption in df['tokenized_caption']]

# Build the LDA model
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, random_state=42, passes=20, iterations=400, eval_every=1)

# Print the topics
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx}: {topic}")

# Calculate coherence score to evaluate the model
coherence_model_lda = CoherenceModel(model=lda_model, texts=df['tokenized_caption'], dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print(f"\nCoherence Score: {coherence_lda}")

# Extract topics from the model
topics = lda_model.show_topics(formatted=False)

# Assign topics to captions
df['topic'] = df['tokenized_caption'].apply(lambda x: sorted(lda_model.get_document_topics(dictionary.doc2bow(x)), key=lambda x: -x[1])[0][0])

# Calculate average likes for each topic
average_likes_per_topic_gensim = df.groupby('topic')['likesCount'].mean()

print(average_likes_per_topic_gensim)

pyLDAvis.enable_notebook()
lda_display = gensimvis.prepare(lda_model, corpus, dictionary, sort_topics=False)
pyLDAvis.display(lda_display)

import matplotlib.pyplot as plt

average_likes_per_topic_gensim.plot(kind='bar', color='skyblue')
plt.xlabel('Topic Number')
plt.ylabel('Average Likes')
plt.title('Average Likes Per Topic')
plt.xticks(rotation=0)
plt.show()
