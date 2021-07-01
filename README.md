# TV-show-character-prediction-using-NLP

This project aims to predicts \"Which Character are You?\" as to which Seinfeld character would be most likely to say the sentence/joke that was entered. The dataset is a text file containing strings of data as dialogues. To solve this problem, I started with data ingestion and performed text mining to get the characters and their dialogues and stored them as a DataFrame. In this process, I applied several transformations in the data to remove the unnecesaary text and the unimportant characters and words. I performed Data Analysis and Topic Modeling to uncover the latent topics.

![1](https://user-images.githubusercontent.com/51110015/124053898-0ccb6700-d9ef-11eb-9dfd-54e435ba81d1.PNG)


Observations: From the above plot, we can observe that there are 4 major characters which lead the show and all other character looks like supporting characters. Jerry is the main character with most dialogues in the show, then george and then elaine and kramer. Followed by these characters there are other characters but their dialogues distribution is very less and seems insignificant.

![2](https://user-images.githubusercontent.com/51110015/124053907-11901b00-d9ef-11eb-9cd7-e022d4b097f1.PNG)



Observation : From the above plot we can observe that, the major dialogues character length is around 1 to 25 and then the number of dialogues decreases as the character length of dialogue increases showing that the converstaion between the charactes are short on a average.

![3](https://user-images.githubusercontent.com/51110015/124053915-1523a200-d9ef-11eb-89a1-d3cb188aa15c.PNG)



Word Cloud is a nice technique to identify important words among others. So I created word cloud of 20 topics and analysised those to understand the topics. Topic 8 shows words related to love with words like miss, relationship, couple. Topic 5 tell Jery or geoge being sick with words related doctor, sick. Topic 18 tells about house/apartment related along with start and anymore. This indicated moving to a new place. Topic 19 talks about george and some troble/problem. Topic 16 tell about hearing of death today and so on.

From the analysis of word cloud some topics look clear but still some are vague. Since sitcom have slang languages and conveys a story rather than any serious scientific well-defined contexts, the topics we got are little vague to understand and are more diffused with very common words. More work could be done to filter out word from the corpus and remove the slang words to obtain a clearer topic modeling.

With the clean data, I used Doc2Vec and Universal Sentence encoder to predict the character for the text entered.

Doc2Vec paragraph embeddings (Baseline model)

My first approach was to used bag of words and TFIDF represention of the dialogues and get the cosine simiarilty between the dialogues. Cosine similarity is a metric used to measure how similar the documents are irrespective of their size. Mathematically, it measures the cosine of the angle between two vectors projected in a multi-dimensional space. The cosine similarity is advantageous because even if the two similar documents are far apart by the Euclidean distance (due to the size of the document), chances are they may still be oriented closer together. The smaller the angle, higher the cosine similarity.

However, tfidf or countvectorizer does not take semantics into account as they are frequency and weight based methods respectively, so, the sematic similarity is not captured using these methods which is not a good approach. When I tried this approach, the observation was that a lot of sentences had highest cosine similarities and one cannot figure out which would be the most similar sentence.

To tackle this issue, semantics needs to be considered. My next approach used is Doc2Vec:

Word2Vec is a more recent model that embeds words in a lower-dimensional vector space using a shallow neural network. The result is a set of word-vectors where vectors close together in vector space have similar meanings based on context, and word-vectors distant to each other have differing meanings. For example, strong and powerful would be close together and strong and Paris would be relatively far.

Gensim’s Word2Vec class implements this model. With the Word2Vec model, we can calculate the vectors for each word in a document. But what if we want to calculate a vector for the entire document, which is a dialogue n our case?

In Gensim, we refer to the Paragraph Vector model as Doc2Vec. The basic idea is: act as if a document has another floating word-like vector, which contributes to all training predictions, and is updated like other word-vectors, but we will call it a doc-vector. The aim is to find semantically similar dialogues to the query sentence for which we want to predict the character. The prediction would be the character whose dialogue matched the most with query vector. The understanding is if a character has spoken a sentence, he/she is most likely to speak a similar sentence.

![doc2vec](https://user-images.githubusercontent.com/51110015/124053934-1e147380-d9ef-11eb-9a37-cd76fd0c4c45.PNG)


**Universal Sentence Encoder**
One of the most well-performing sentence embedding techniques right now is the Universal Sentence Encoder by Google. The key feature is we can use it for multitask learning. This means that the sentence embeddings we generate can be used for multiple tasks like sentiment analysis, text classification, sentence similarity, etc, and the results of these tasks are then fed back to the model to get even better sentence vectors that before.

The most interesting part is that this encoder is based on two encoder models and we can use either of the two:

Transformer
Deep Averaging Network(DAN)
There is a trade-off between the accuracy of the results obtained and the resources required for computation, when we compare both the variants. The transformer-based model aims to achieve high model accuracy, but it requires a high amount of computation resources and increases model complexity. The memory usage and computation time for this variant rise erratically with the length of the sentence. On the contrary, the computation time linearly increases with sentence length for the DAN-based model. In the research paper, the transformer model’s time complexity has been noted as O(n2)while that of DNA model as O(n), where ‘n’ denotes the sentence length. The DAN variant aims at efficient inference despite a little reduction in achieved accuracy. Both of these models are capable of taking a word or a sentence as input and generating embeddings for the same. I am using DAN architechtuer for my implementation. The following is the basic flow:

Tokenize the sentences after converting them to lowercase
Depending on the type of encoder, the sentence gets converted to a 512-dimensional vector. If we use the transformer, it is similar to the encoder module of the transformer architecture and uses the self-attention mechanism. The DAN option computes the unigram and bigram embeddings first and then averages them to get a single embedding. This is then passed to a deep neural network to get a final sentence embedding of 512 dimensions.
These sentence embeddings are then used for various unsupervised and supervised tasks and here I am using it for sentence semantic similarity. The trained model is then again reused to generate a new 512 dimension sentence embedding.

![USE](https://user-images.githubusercontent.com/51110015/124053950-22d92780-d9ef-11eb-836b-1afc7f1256c7.PNG)
