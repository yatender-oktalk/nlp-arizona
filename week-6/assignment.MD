# Assignment 
## I. True or False

1. **True** - The spaCy library can preprocess documents in many languages, including Spanish.
2. **True** - In Keras, a TimeDistributed layer is indeed a wrapper that allows a recurrent layer to return an output at every time step.
3. **False** - Sorting vocabulary alphanumerically versus randomly affects the indices assigned to each word, changing the one-hot encoded vectors.
4. **True** - A common way to represent out-of-vocabulary (OOV) words in one-hot encoding is by using a zero vector.
5. **True** - It is technically possible to represent 10,000 words with 10-dimensional embeddings, although such embeddings might not capture the complexity well.
6. **False** - A predicted value of 0.43, assuming a threshold of 0.5 for classification, would typically classify the article as not containing misinformation.
7. **False** - If all examples are positive, the cross-entropy loss would be zero because the predicted probabilities perfectly match the actual labels.
8. **False** - A bi-directional RNN is not necessarily more efficient in terms of parameters; it generally requires more parameters because it processes sequences in both directions.
9. **True** - If the forget gate outputs zeros, it indicates that the LSTM should forget all information currently stored in the cell state.
10. **False** - Generally, both input and output sequences need to be padded/truncated to handle different lengths effectively in sequence-to-sequence models.
11. **True** - Contextual embeddings, like those from BERT or GPT, can capture different meanings of the same word in different contexts (polysemy).
12. **False** - Pre-trained models can be fine-tuned on multiple tasks sequentially.
13. **False** - Sparse self-attention mechanisms are designed to reduce the computational requirements from quadratic to more manageable levels.
14. **False** - Training of models like ChatGPT involves human supervision during the initial training phase, often in the form of selecting training data and designing model architectures.
15. **False** - A sigmoid function accepts any real number as input but restricts the output to the range between 0 and 1.

## II. Fill in the blanks

1. **The _adjudicator_** is in charge of resolving discrepancies among the annotations produced by the annotators.
2. **Stop words** are words that can be filtered out from textual data because they are so frequent that they provide little information.
3. You have the sentence "Time flies when you're having fun." tokenized by words and annotated with Part-of-Speech. To represent this annotation following the BIO schema, there should be **five** tokens with the label O.
4. A n-gram language model that only attends to the previous word in the sequence is called a **bigram** language model.
5. The dot product of 2 word-embeddings is 10. If their magnitudes were 5 and 4, their cosine similarity would be **0.5**.
6. You are working on a text classification problem with 3 classes, and you have implemented a model with a softmax in the output layer. For a specific input, the model returns the following probabilities: 0.31, 0.14, and **0.55**.
7. A character-based tokenization of the sentence "Can't wait, it's almost vacation time." would result in **32** tokens.
8. The maximum value of BLEU's Brevity Penalty is **1**.
9. **Attention** allows a deep-learning model to selectively focus on certain parts of the input sequence based on the relevance of each token to the others.
10. Given the following confusion matrix for a sentiment analysis model, the macro-average F1 score needs to be calculated based on true positives, false positives, false negatives, and true negatives for each class separately and then averaged. Answer is **0.878**.

## Multiple Choice questions. 

1. What AutoClass of the transformers library could be used to instantiate a pre-trained model for a sequence labeling task?
   - **a. AutoModelForTokenClassification**

2. Which of the following pre-processing steps should always be taken?
   - **a. None, it depends on the task.**

3. A word that is very frequent in a document but very infrequent in the rest of the documents in a corpus will have:
   - **c. Low tf and high idf.**

4. Given the co-occurrence probabilities ratio p(w1|w2)/p(w1|w3) = 10, GloVe will learn embeddings for w1, w2, and w3 such that:
   - **a. w1 and w2 are closer together than w1 and w3.**

5. The range of the output values of a relu function is:
   - **a. [0, ∞)**

6. A model for Named Entity Recognition is able to identify all the entities in a test set, however the model is only able to predict one token per entity. For example, for the named entity "Frida Kahlo", the model only identifies "Frida". Using a relaxed evaluation, the precision of the model would be:
   - **c. 0.5**

7. Both GTP and BERT are based on the transformer architecture, but they only use part of it:
   - **a. GPT uses the decoder and BERT uses the encoder.**

8. Which of the following statements about the Embedding layer is not true?
   - **d. It can be a square matrix.**

9. Which of the following NLP approaches is most suitable for sentence segmentation?
   - **a. Sequence Labeling**

10. Which of the following corruptions of the tokenized input "SGD is an optimizer. It learns from errors." would not be used for pre-training BART?
    - **a. "SGD is an error. It learns from optimizer."**

## Short Answer

1. We need to train `TfidfVectorizer` of scikit-learn on the training data because it needs to learn the idf component of the vectors, which depends on the frequency of each term in the training corpus.

2. Annotation guidelines should include the purpose of the annotation, detailed instructions on how to annotate, definitions of categories or tags, examples of correct and incorrect annotations, and procedures for handling ambiguous cases.

3. To make Beam Search behave like Greedy Search, set the beam width to 1, which makes the algorithm choose the single best option at each step without considering broader possibilities.

4. During BERT pre-training, the [CLS] token is used to aggregate the representation of the sequence which is then used in tasks such as next sentence prediction.

5. To distinguish training examples from different tasks when pre-training a multitask seq-to-seq model, one can use task-specific prefixes or embed a task identifier into the input sequences.
