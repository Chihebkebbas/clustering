from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import os
import pandas as pd, numpy as np
from numpy.linalg import norm

def preprocess(text_data, option):
    """Preprocess textual data to have numerical vectors

    Args:
        text_data: input textual data as a data frame
        option: type of encoder to transform categorical data
    """
    if option == "BOW":
        # TODO: use CountVectorizer to have one-hot encoding to represent bag-of-words
        vectorizer = CountVectorizer(lowercase=True)
        vectorizer.fit(text_data)
        vectors = vectorizer.transform(text_data)

        return vectors.toarray()

    else:
        # TODO: use SentenceTransformer to have embedding vectors
        model = SentenceTransformer(option)
        vectors = model.encode(text_data)
        return vectors



def test_preprocess(option):
    """Test preprocessing methods on similar and dissimilar English sentences

    Args:
        option: type of encoder to transform categorical data
    """
    print('>test du prétraitement')
    data=["She had welcomed my unexpected return as a blessing from heaven.",
    "She had welcomed my return.",
    "She had perceived my unanticipated return as a divine intervention.",
    "She was very happy to see me come back.",
    "She hates my cat, which is unexpected.",
    "She drinks beers, which is a bad habit."]
    prep_data = preprocess(data,option)
    print(">>>Phrase de départ :",data[0])
    for i in range(1,len(data)):
        print(">>>similarité (échelle de 0 à 1) avec: ", data[i])
        print(">>>\t", np.dot(prep_data[0],prep_data[i])/(norm(prep_data[0])*norm(prep_data[i])))
        print()

test_preprocess("thenlper/gte-small")