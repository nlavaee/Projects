import sys
import os
import math
import numpy as np
import itertools
import re, string
from collections import defaultdict, OrderedDict
import operator
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt



def generateBigrams(inputString):
    charac_bigrams = []
    charac_bigrams = ([inputString[i:i+2] for i in range(len(inputString)-1)])
    # print(charac_bigrams)
    return charac_bigrams
    # inputString.split()
    # print(inputString.split())
    # return inputStrin.split()

def generateTrigrams(inputString):
    n = 3
    charac_trigrams = []
    charac_trigrams = ([inputString[i:i+n] for i in range(len(inputString)-n+1)])
    return charac_trigrams

def calcPosteriors(labels, priors, features, feature_test, posteriors):
    max_labels = defaultdict()
    best_label = ''
    for doc, word in feature_test.items():
        curr_max = float("-inf")
        for label, files in labels.items():
            # print("label is ", label)
            # print("files are ", files)
            posterior_sum = 0
            prior = math.log(priors[label], 2)

            for item in features:
                if word[item] != 0:
                    posterior_sum += math.log(posteriors[label][item], 2)
                # elif word[item] == 0:
                #     # print("also got here")
                #     posterior_sum += math.log((1 - posteriors[label][item]), 2)

            # print("curr max is ", curr_max, '\n')
            # print("prior and posterior is ", prior + posterior_sum)
            # print("current doc is ", doc)
            # print("current label is ", label)
            if(curr_max < (prior + posterior_sum)):
                curr_max = prior + posterior_sum
                best_label = label
                # print("best label is ", best_label, '\n')

        max_labels[doc] = best_label
        # top_labels.append(best_label)

    max_labels = OrderedDict(sorted(max_labels.items()))
    # print(posteriors["c02"])
    # print(max_labels)
    return max_labels

def calcAccuracy(max_labels, truths):
    # garbage, problem = (sys.argv[1].split("/problem"))

    correct_labels = float(0.0)
    accuracy = float(0.0)
    for key, val in max_labels.items():
        for key1, val1 in truths.items():
            if val == val1:
                correct_labels += 1

    accuracy = (correct_labels/len(max_labels))

    # print(accuracy)
    return accuracy

def add_all_features(feature_freq_label, all_features):
    for language, feature_list in feature_freq_label.items():
        for item in all_features:
            if item not in feature_list:
                feature_freq_label[language][item] = 0

def calc_loglikelihood(unique_features, posteriors, feature_freq_label):
    for label, feature_list in feature_freq_label.items():
        conditional_indepen = dict()
        for feature in feature_list:
            denom_1 = 0
            numerator = feature_freq_label[label][feature] + 1
            for uniq_ft in unique_features:
                denom_1 += feature_freq_label[label][uniq_ft]

            denom = len(unique_features) + denom_1
            conditional_indepen[feature] = numerator/denom


        posteriors[label] = conditional_indepen


def gen_test_feat(test_words, all_features, feature_test):
    for doc, words in test_words.items():
        # print(len(words), '\n')
        feature_vec = defaultdict()
        for item in all_features:
            if item in words:
                feature_vec[item] = 1
            else:
                feature_vec[item] = 0
        #   print(item, '\n')
            # feature_vec[item] = sum(item in s for s in words)
        feature_test[doc] = feature_vec

def calc_priors(priors, label_docs, total_numb_trainingdocs):
    for label, docs in label_docs.items():
        priors[label] = len(docs)/total_numb_trainingdocs

def create_confusion_matrix(label_docs, max_labels, ground_truth):
    numb_labels = len(label_docs)

    predicted_language = []
    ground_truth_language = []

    for key, val in max_labels.items():
        if val == 'English':
            predicted_language.append(1)
        elif val == 'French':
            predicted_language.append(2)
        elif val == 'German':
            predicted_language.append(3)
        elif val == 'Spanish':
            predicted_language.append(4)

    for key, val in ground_truth.items():
        if val == 'English':
            ground_truth_language.append(1)
        elif val == 'French':
            ground_truth_language.append(2)
        elif val == 'German':
            ground_truth_language.append(3)
        elif val == 'Spanish':
            ground_truth_language.append(4)


    return numb_labels, predicted_language, ground_truth_language


def computeConfusionMatrix(predicted, groundTruth, nAuthors):
    confusionMatrix = [[0 for i in range(nAuthors+1)] for j in range(nAuthors+1)]

    for i in range(len(groundTruth)):
        confusionMatrix[predicted[i]][groundTruth[i]] += 1

    return confusionMatrix

def outputConfusionMatrix(confusionMatrix):
    columnWidth = 4

    print(str(' ').center(columnWidth),end=' ')
    for i in range(1,len(confusionMatrix)):
        print(str(i).center(columnWidth),end=' ')

    print()

    for i in range(1,len(confusionMatrix)):
        print(str(i).center(columnWidth),end=' ')
        for j in range(1,len(confusionMatrix)):
            print(str(confusionMatrix[j][i]).center(columnWidth),end=' ')
        print()


def main():

    # ----- Generate data structures and data for bi-grams -----

    print("Calculating accuracy and confusion matrix for bi-grams...")

    temp_features = []
    test_data = []
    labels = defaultdict(list)
    train_words = defaultdict(list)
    test_words = defaultdict(list)
    ground_truth = defaultdict()
    total_numb_trainingdocs = 0
    label_docs = defaultdict(list)
    all_features = []

    feature_freq = defaultdict(int)
    feature_freq_label = defaultdict(dict)

    directory = os.getcwd() + "/" + sys.argv[1] + "/"


    for file in os.listdir(directory):
    
        words = []
        test_data = []
        # print(file)
        if(not file.endswith(".txt")):
            with open(os.path.join(directory, file), encoding="iso-8859-15", errors='ignore') as inputFile1:
                for line in inputFile1:
                    # print(line)

                    # --- bigrams ---
                    line = line.lower()
                    temp_features = (generateBigrams(line.rstrip()))
                    all_features.extend(generateBigrams(line.rstrip()))
                    #---bigrams-----

                    # temp_features = generateTrigrams(line.rstrip())
                    # all_features.extend(generateTrigrams(line.rstrip()))
                    

                    line = line.rstrip()
                    line = re.findall('..', line)
                    # line = (line.rstrip())
                    words.extend(line)

                    for item in temp_features:
                        feature_freq[item] += line.count(item)

                    # features = []

            garbage, language = file.split("-")
            language, garbage = language.split(".")
            labels[language].append(file)
            train_words[language] = words
            total_numb_trainingdocs += 1
            label_docs[language].append(file)
            feature_freq_label[language] = feature_freq
            feature_freq = defaultdict(int)


        else:
            with open(os.path.join(directory, file), encoding="utf-8", errors='ignore') as inputFile2:
                for line in inputFile2:
                    line = line.rstrip()
                    line = line.lower()
                    #---bigrams----
                    line = re.findall('..', line)
                    #---bigrams----
                    # line = re.findall('...', line)
                    test_data.extend(line)


            garbage, language = file.split("-")
            language, garbage = language.split(".")
            test_words[file] = test_data
            ground_truth[file] = language

    # -----------------------------------------------

    # -------- Call functions for bi-grams ----------

    add_all_features(feature_freq_label, all_features)

    unique_features = list(dict.fromkeys(all_features))

    posteriors = defaultdict(dict)

    # print("GOT HERE1")

    calc_loglikelihood(unique_features, posteriors, feature_freq_label)


    feature_test = defaultdict(list)


    gen_test_feat(test_words, all_features, feature_test)


    priors = defaultdict()

    # calculate the priors
    calc_priors(priors, label_docs, total_numb_trainingdocs)
   

    # print(label_docs)
    label_docs = OrderedDict(sorted(labels.items()))
    priors = OrderedDict(sorted(priors.items()))

    max_labels = calcPosteriors(label_docs, priors, all_features, feature_test, posteriors)

    # print(max_labels)

    ground_truth = OrderedDict(sorted(ground_truth.items()))

    accuracy = calcAccuracy(max_labels, ground_truth)

    print("Accuracy for bi-grams is ", accuracy)

    numb_labels, predicted_language, ground_truth_language = create_confusion_matrix(label_docs, max_labels, ground_truth)

    confusion_matrix = computeConfusionMatrix(predicted_language, ground_truth_language, numb_labels)

    print("Confusion Matrix: 1 = 'English', 2 = 'French', 3 = 'German', 4 = 'Spanish'")

    outputConfusionMatrix(confusion_matrix)



    # ----------------------------------------------------

    print("Calculating accuracy and confusion matrix for tri-grams...")

    # ----- Generate data structures and data for tri-grams -----

    temp_features = []
    test_data = []
    labels = defaultdict(list)
    train_words = defaultdict(list)
    test_words = defaultdict(list)
    ground_truth = defaultdict()
    total_numb_trainingdocs = 0
    label_docs = defaultdict(list)
    all_features = []

    feature_freq = defaultdict(int)
    feature_freq_label = defaultdict(dict)

    directory = os.getcwd() + "/" + sys.argv[1] + "/"


    for file in os.listdir(directory):
    
        words = []
        test_data = []
        # print(file)
        if(not file.endswith(".txt")):
            with open(os.path.join(directory, file), encoding="iso-8859-15", errors='ignore') as inputFile1:
                for line in inputFile1:
                    # print(line)

                    # --- bigrams ---
                    # temp_features = (generateBigrams(line.rstrip()))
                    # all_features.extend(generateBigrams(line.rstrip()))
                    #---bigrams-----

                    temp_features = generateTrigrams(line.rstrip())
                    all_features.extend(generateTrigrams(line.rstrip()))
                    

                    line = line.rstrip()
                    line = re.findall('...', line)
                    # line = (line.rstrip())
                    words.extend(line)

                    for item in temp_features:
                        feature_freq[item] += line.count(item)

                    # features = []

            garbage, language = file.split("-")
            language, garbage = language.split(".")
            labels[language].append(file)
            train_words[language] = words
            total_numb_trainingdocs += 1
            label_docs[language].append(file)
            feature_freq_label[language] = feature_freq
            feature_freq = defaultdict(int)


        else:
            with open(os.path.join(directory, file), encoding="utf-8", errors='ignore') as inputFile2:
                for line in inputFile2:
                    line = line.rstrip()
                    #---bigrams----
                    # line = re.findall('..', line)
                    #---bigrams----
                    line = re.findall('...', line)
                    test_data.extend(line)


            garbage, language = file.split("-")
            language, garbage = language.split(".")
            test_words[file] = test_data
            ground_truth[file] = language

    # -----------------------------------------------

    # -------- Call functions for tri-grams ----------

    add_all_features(feature_freq_label, all_features)


    unique_features = list(dict.fromkeys(all_features))

    posteriors = defaultdict(dict)

    # print("GOT HERE1")

    calc_loglikelihood(unique_features, posteriors, feature_freq_label)


    feature_test = defaultdict(list)


    gen_test_feat(test_words, all_features, feature_test)


    priors = defaultdict()

    # calculate the priors
    calc_priors(priors, label_docs, total_numb_trainingdocs)
   

    # print(label_docs)
    label_docs = OrderedDict(sorted(labels.items()))
    priors = OrderedDict(sorted(priors.items()))

    max_labels = calcPosteriors(label_docs, priors, all_features, feature_test, posteriors)

    # print(max_labels)

    ground_truth = OrderedDict(sorted(ground_truth.items()))

    accuracy = calcAccuracy(max_labels, ground_truth)

    # print(ground_truth)

    # print(max_labels)

    print("Accuracy for trigrams is ", accuracy)

    numb_labels, predicted_language, ground_truth_language = create_confusion_matrix(label_docs, max_labels, ground_truth)

    confusion_matrix = computeConfusionMatrix(predicted_language, ground_truth_language, numb_labels)

    print("Confusion Matrix: 1 = 'English', 2 = 'French', 3 = 'German', 4 = 'Spanish'")

    outputConfusionMatrix(confusion_matrix)

    # ----------------------------------------------------

    

if __name__ == '__main__':
    main()