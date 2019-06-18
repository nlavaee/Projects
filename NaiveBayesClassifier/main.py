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
                elif word[item] == 0:
                    posterior_sum += math.log((1 - posteriors[label][item]), 2)

            # print("curr max is ", curr_max, '\n')
            # print("prior and posterior is ", prior + posterior_sum)
            # print("current doc is ", doc)
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
    garbage, problem = (sys.argv[1].split("/problem"))

    correct_labels = float(0.0)
    accuracy = float(0.0)
    for key, val in max_labels.items():
        if val == truths[(key, problem)]:
            correct_labels += 1

    accuracy = (correct_labels/len(max_labels))

    # print(accuracy)
    return accuracy

def createConfusionMatrix(max_labels, truths, labels):
    garbage, problem = (sys.argv[1].split("/problem"))

    predictedAuthorIdNum = []
    groundTruthAuthorIdNum = []
    nAuthors = len(labels)

    for val in max_labels.values():
        garbage, author = val.split("r")
        predictedAuthorIdNum.append((int(author)))

    for key, val in truths.items():
        if key[1] == problem and val != "__NONE__":
            garbage, author = val.split("r")
            groundTruthAuthorIdNum.append((int(author)))


    outputConfusionMatrix(computeConfusionMatrix(predictedAuthorIdNum, groundTruthAuthorIdNum, nAuthors))

def calcStopwordFreq(features, train_words):
        word_frequencies = defaultdict()

        for item in features:
            word_frequencies[item] = 0

        for doc, words in train_words.items():
            for item in features:
                word_frequencies[item] += words.count(item)


        word_frequencies = {k: v for k, v in sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)}
        # print(word_frequencies)
        return word_frequencies


def calcFeatureCurve(word_frequencies, labels, priors, feature_test, posteriors, truths):

    top_accuracies = [] # x axis
    numb_features = [] # y axis
    top_features = []

    for count, item in enumerate(word_frequencies, 1):

        if count%10 == 0:
            top_features = (({k: word_frequencies[k] for k in list(word_frequencies)[:count]}))

            top_labels = calcPosteriors(labels, priors, top_features, feature_test, posteriors)

            accuracy = calcAccuracy(top_labels, truths)

            top_accuracies.append(accuracy)

            numb_features.append(count)


    return top_accuracies, numb_features

def calcCCE(features, labels, priors, posteriors):
    CCE = defaultdict()

    for feature in features:
        CCE_i = 0
        for label, doc in labels.items():
            prior_prob = priors[label]
            posterior_prob = posteriors[label][feature]
            log_posterior_prob = math.log(posteriors[label][feature], 2)

            CCE_i += (prior_prob*posterior_prob*log_posterior_prob)*-1

        CCE[feature] = CCE_i

    CCE = {k: v for k, v in sorted(CCE.items(), key=lambda x: x[1], reverse=True)}
    # print(CCE)
    top20entropy = {k: CCE[k] for k in list(CCE)[:20]}
    # print(top20entropy)
    return top20entropy


def stripWhitespace(inputString):
    return re.sub("\s+", " ", inputString.strip())

def tokenize(inputString):
    # print(inputString)
    whitespaceStripped = stripWhitespace(inputString)
    punctuationRemoved = "".join([x for x in whitespaceStripped
                                  if x not in string.punctuation])
    lowercased = punctuationRemoved.lower()
    return lowercased.split()


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

    # read in ground truth values for testing

    ground_truth_directory = os.getcwd() + "/"

    ground_truths = defaultdict()

    # problem_list = ["A" , "B", "C", "D", "E", 
    # "F", "G", "H", "I", "J", "K", "L", "M"]
    with open(os.path.join(ground_truth_directory, "test_ground_truth.txt")) as f:
        # for problem in problem_list:
        truths = defaultdict()
        # next_problem = 0
        for line in f:
            if(line != '\n'):
                problem, garbage = line.split("/")
                # print(problem[-1])
                # garbage, problem = (sys.argv[1].split("/problem"))
                garbage, info = line.split("/")
                file, author = info.split(' ')
                truths[(file, problem[-1])] = author.rstrip()


    directory = os.getcwd() + "/" + sys.argv[1] + "/"
    # print(directory)

    # print(path)
    features = [] # these are the stopwords
    stop_words_directory = os.getcwd()  + "/read_stopwords/"
    with open(os.path.join(stop_words_directory, "stopwords.txt")) as f:
        for line in f:
            if(line != '\n'):
                # if len(line) - line.count(' ') != 2:
                features.append(line.rstrip())
                    # print(line)
                    # print("length of the line is ", len(line) - line.count(' '), '\n')
                    # continue
                # else:
                #     features.append(line)

    # print(features)

    # remove duplicates from a list
    features = list(dict.fromkeys(features))

    # separate out training exampels from test examples
    train_words = defaultdict(list)
    test_words = defaultdict(list)

    # training_docs = defaultdict(dict)

    labels = defaultdict(list)

    total_docs_training = 0

    for file in os.listdir(directory):
        words = []
        test = []
        with open(os.path.join(directory, file), encoding='utf-8', errors='ignore') as inputFile:
            for line in inputFile:
                words.extend(tokenize(line))
                test.extend(line.rstrip())


        if "sample" in file:
            test_words[file] = words

        else:
            total_docs_training += 1
            garbage, num = file.split('n')
            num, garbage = num.split('-')
            
            labels['Author'+ num].append(file)
            train_words[file] = words



    # print(train_words) 
    # print(labels)

    # train

    # train_results = defaultdict(dict)

    # labels_keys = dict((v,k) for k,v in labels.items())

    stop_word_freq = defaultdict(dict)

    for doc, words in train_words.items():
        stop_words = defaultdict()
        # print(words)
        for item in features:
            stop_words[item] = words.count(item)

        stop_word_freq[doc] = (stop_words)


   
   
    posteriors = defaultdict(dict)
    # posterior_not_in = defaultdict(list)

    # print(stop_word_freq)

    # print(stop_word_freq)

    for label, files in labels.items():
        Nc = len(files)
        # print("the label is ", label, '\n')
        # print("the files are ", files, '\n')
        conditional_indepen = dict()
        for item in features:
            Nci = 0
            for file in files:
                if stop_word_freq[file][item] != 0:
                    Nci += 1

            conditional_indepen[item] = ((Nci + 1)/(Nc + 2))

        posteriors[label] = conditional_indepen


    # print(posteriors)


    # print(posteriors)

    priors = defaultdict()

    for label, files in labels.items():
        # print("the files are ", files)
        prior_prob = len(files)/total_docs_training
        priors[label] = prior_prob


    # print(priors)

    # print(features)
    feature_test = defaultdict(dict)
    list_test_feat = []

    for doc, word in test_words.items():
        feature_vec = dict()
        for item in features:
            feature_vec[item] = word.count(item)
            # print("the number of times feature ", item, "shows up is ", word.count(item), "for doc ", doc)
        # print("feature vec is ", feature_vec)
            # print("number of times word ", item, "shows up is ", word.count(item), " in doc ", doc,  '\n')

        feature_test[doc] = (feature_vec)
        list_test_feat.append(feature_vec)


    # print(feature_test)

    
    # print(labels)

    # TEST

    # sort the labels so that you select the smallest label in case of tie
    labels = OrderedDict(sorted(labels.items()))
    priors = OrderedDict(sorted(priors.items()))
    # top_labels = []

    # PUT FUNCTION CALC_POSTERIORS HERE 

    max_labels = calcPosteriors(labels, priors, features, feature_test, posteriors)


    # print(feature_test)

    # calculate accuracy
    print("Accuracy: ")
    print("-----------")
    accuracy = calcAccuracy(max_labels, truths)
    print(accuracy)
    # output confusion matrix

    print("Confusion Matrix:")
    print("-------------------")
    createConfusionMatrix(max_labels, truths, labels)

    # print(training_docs)

    # Feature curve

    word_frequencies = calcStopwordFreq(features, train_words)


    top_accuracies, numb_features =  calcFeatureCurve(word_frequencies, labels, priors, feature_test, posteriors, truths)

    # Feature Ranking


    top_features = calcCCE(features, labels, priors, posteriors)

    print("Top Features:")
    print("---------------")


    for key, val in top_features.items():
        print(key, ": ", val)


    print("Training w/ Frequent Features")
    print("-------------------------------")
    for i in range(len(top_accuracies)):
        print(numb_features[i], ": ", top_accuracies[i])


    plt.plot(numb_features, top_accuracies, '-')
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Top 10N Features")
    plt.title("Accuracy vs. Top Features")
    plt.show()

    # print(test)



    # print(max_labels)
    # print(truths)



if __name__ == '__main__':
    main()