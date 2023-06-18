# Importing libraries
import nltk
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# download the treebank corpus from nltk
nltk.download('treebank')

# download the universal tagset from nltk
nltk.download('universal_tagset')

# reading the Treebank tagged sentences
nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))

# split data into training and validation set in the ratio 80:20
train_set, test_set = train_test_split(nltk_data, train_size=0.9, test_size=0.1, random_state=101)

# create list of train and test tagged words
train_tagged_words = [tup for sent in train_set for tup in sent]
test_tagged_words = [tup for sent in test_set for tup in sent]
print(len(train_tagged_words))
print(len(test_tagged_words))

# use set datatype to check how many unique tags are present in training data
tags = {tag for word, tag in train_tagged_words}
print(len(tags))
print(tags)

# # check total words in vocabulary


# compute Emission Probability
def word_given_tag(word, tag, train_bag=train_tagged_words):
    tag_list = [pair for pair in train_bag if pair[1] == tag]
    count_tag = len(tag_list)  # total number of times the passed tag occurred in train_bag
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0] == word]
    # now calculate the total number of times the passed word occurred as the passed tag.
    count_w_given_tag = len(w_given_tag_list)

    return count_w_given_tag, count_tag


# compute  Transition Probability
def t2_given_t1(t2, t1, train_bag=train_tagged_words):
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t == t1])
    count_t2_t1 = 0
    for index in range(len(tags) - 1):
        if tags[index] == t1 and tags[index + 1] == t2:
            count_t2_t1 += 1
    return count_t2_t1, count_t1


# creating t x t transition matrix of tags, t= no of tags
# Matrix(i, j) represents P(jth tag after the ith tag)

tags_matrix = np.zeros((len(tags), len(tags)), dtype='float32')
for i, t1 in enumerate(list(tags)):
    for j, t2 in enumerate(list(tags)):
        tags_matrix[i, j] = t2_given_t1(t2, t1)[0] / t2_given_t1(t2, t1)[1]

print(tags_matrix)

# convert the matrix to a df for better readability
# the table is same as the transition table shown in section 3 of article
tags_df = pd.DataFrame(tags_matrix, columns=list(tags), index=list(tags))
print(tags_df)


def Viterbi(words, train_bag=train_tagged_words):
    state = []
    T = list(set([pair[1] for pair in train_bag]))

    for key, word in enumerate(words):
        # initialise list of probability column for a given observation
        p = []
        for tag in T:
            if key == 0:
                transition_p = tags_df.loc['.', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]

            # compute emission and state probabilities
            emission_p = word_given_tag(words[key], tag)[0] / word_given_tag(words[key], tag)[1]
            state_probability = emission_p * transition_p
            p.append(state_probability)

        pmax = max(p)
        # getting state for which probability is maximum
        state_max = T[p.index(pmax)]
        state.append(state_max)
    return list(zip(words, state))


test_tagged_words = [tup for sent in test_set for tup in sent]
test_untagged_words = [tup[0] for sent in test_set for tup in sent]
test_tags = [tup[1] for sent in test_set for tup in sent]

start = time.time()
tagged_seq = Viterbi(test_untagged_words)
end = time.time()
difference = end - start

print("Time taken in minutes: ", difference / 60)

predicted_tags = [i[1] for i in tagged_seq]

# accuracy
accuracy = accuracy_score(test_tags, predicted_tags)

print('Viterbi Algorithm Accuracy: ', accuracy * 100)
