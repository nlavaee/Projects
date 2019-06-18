#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cstring>
#include <map>
#include <set>
#include <cmath>
#include <iomanip>
#include <sstream>
#include "csvstream.h"

using namespace std;

class Classifier {
    
public:
    
    //EFFECTS: Returns a set containing the unique "words" in the original
    //string, delimited by whitespace
    static set<string> unique_words(const string &str) {
        istringstream source(str);
        set<string> words;
        string word;
        
        while (source >> word) {
            words.insert(word);
        }
        return words;
    }
    
    //EFFECTS: Probability of label C and is a reflection of how common it is
    double log_prior(double numb_words_labelC, double numb_posts) {
        return log(numb_words_labelC / numb_posts);
    }
    
    //EFFECTS: log-likelihood of a word w given a label C
    double log_likelihood(double num_words_labelC, double num_words_labelC_w) {
        return log(num_words_labelC_w / num_words_labelC);
    }
    
    //EFFECTS: probablilty that w will never be seen in the post with label C
    double never_see_w(double numb_words_w, double numb_posts) {
        return log(numb_words_w / numb_posts);
    }
    
    //EFFECTS: probability that w not in the training set at all
    //so not in content at all
    double no_w(double numb_posts) {
        return log(1 / numb_posts);
    }
    
    //EFFECTS: inserts the unique words and the label into a map
    void words_and_labels(map<string, string> &m, map<string, double>&word_count,
                          map<string, double>&label_count, map<string, map<string, double>>& probability) {
        label_count[m["tag"]]++;
        vocab = unique_words(m["content"]);
        for (auto &word : vocab) {
            word_count[word]++;
            probability[m["tag"]][word]++;
        }
    }
    
    //EFFECTS: prints output for the training data
    void print_classes(map<string, double>&label_count,
                       double total_numb_posts) {
        for (auto &label : label_count) {
            double log_prob = log_prior(label_count[label.first],
                                        total_numb_posts);
            double temp = log_prob;
            
            cout << "  " << label.first << ", " << int(label.second)
            << " examples, log-prior = " << temp << endl;
            
        }
        
    }
    
    //EFFECTS: Prints out the classifier information for the train data
    void classify_train(map<string, double>&label_count, map<string, double>&word_count,
                        map<string, map<string, double>>&probability, double total_numb_posts, int argc, char*argv[]) {
        for (auto &label : label_count) {
            for (auto &word : word_count) {
                if (probability[label.first][word.first] != 0) {
                    double log_like = log_likelihood(label_count[label.first], probability[label.first][word.first]);
                    if (word.second != 0) {
                        if (argc == 4 && strcmp(argv[2], "--debug")) {
                            cout << "  " << label.first << ":" << word.first << ", count = "
                            << probability[label.first][word.first] << ", log-likelihood = " << log_like << endl;
                        }//end if (argc == 4...
                    }//end if (word.second..
                }//end if(probability...
            }//end inner loop
        }//end outer loop
    }
    
    
    //EFFECTS: Prints out the classifier information for the train data and returns performance info
    int classify_test(map<string, double>&label_count, set<string>&uniq_words, map<string, map<string,
                      double>>&probability, double total_numb_posts,
                      map<string, double>&word_count, map<string, string>map_one, int &performance) {
        
        string max_label;
        double largest = -std::numeric_limits<double>::infinity();
        
        for (auto &label : label_count) {
            double log_probability = log_prior(label_count[label.first], total_numb_posts);
            //does the different computations needed to predict the label
            for (auto &word : uniq_words) {
                if (probability[label.first][word] != 0) {
                    log_probability += log_likelihood(label_count[label.first], probability[label.first][word]);
                }
                else if (word_count.count(word) != 0) {
                    log_probability += never_see_w(word_count[word], total_numb_posts);
                    
                }
                else {
                    log_probability += no_w(total_numb_posts);
                    
                }
            }
            //makes sure to get the lable with the highest probability
            if (log_probability > largest) {
                largest = log_probability;
                max_label = label.first;
            }
        }
        cout << " predicted = " << max_label << ", log-probability score = " << largest << endl;
        
        //if the correct tag matches the perdicted label, increment performance
        if (map_one["tag"] == max_label) {
            ++performance;
        }
        return performance;
    }
    
    
private:
    
    std::set<string> vocab;
    
};

int main(int argc, char*argv[]) {
    
    cout << setprecision(3);
    
    string filename1 = string(argv[1]);
    string filename2 = string(argv[2]);
    
    if ((argc == 4 && strcmp(argv[2], "--debug")) || (argc == 3)) {
        set<string> w;
        set<string> content;
        set<string> total_numb_uniqwords;
        set<string> tags;
        
        double total_numb_posts = 0;
        
        Classifier c;
        //declare these as global variables so you can work with them
        std::map<string, double> word_count;
        std::map<string, double> label_count;
        std::map<string, std::map<string, double>> probability;
        
        
        if (argc == 4 && strcmp(argv[2], "--debug")) {
            cout << "training data:" << endl;
        }
        
        try {
            csvstream csvin(argv[1]);
            
            map<string, string>map;
            // int post_number = 0;
            
            //std::map<string, map<string, double>> probability;
            while (csvin >> map) {
                
                //total number of training sets
                ++total_numb_posts;
                //takes unique words from each row
                w = c.unique_words(map["content"]);
                for (auto &uniq : w) {
                    //loop through the set w, and insert all the words inside of other set
                    total_numb_uniqwords.insert(uniq);
                }
                //insert the content
                content.insert(map["content"]);
                tags.insert(map["tag"]);
                
                
                if (argc == 4 && strcmp(argv[2], "--debug")) {
                    cout << "  label = " << map["tag"] << ", content = " << map["content"] << endl;
                }
                c.words_and_labels(map, word_count, label_count, probability);
            }
            //calculate the probability
            
            
            double vocab_size = total_numb_uniqwords.size();
            if ((argc == 4 && strcmp(argv[2], "--debug")) || (argc == 3)) {
                cout << "trained on " << int(total_numb_posts) << " examples" << endl;
            }
            
            if (argc == 4 && strcmp(argv[2], "--debug")) {
                cout << "vocabulary size = " << int(vocab_size) << endl << endl;
                
                cout << "classes:" << endl;
                //put here
                c.print_classes(label_count, total_numb_posts);
            }
            if (argc == 4 && strcmp(argv[2], "--debug")) {
                cout << "classifier parameters:" << endl;
            }
            //put classifier here
            
            c.classify_train(label_count, word_count, probability, total_numb_posts, argc, argv);
            
        }
        catch (csvstream_exception &e) {
            cout << "Error opening file: " << filename1 << endl;
            exit(EXIT_FAILURE);
        }
        
        try {
            csvstream csvin1(argv[2]);
            std::map<string, string> map_one;
            int num_posts_test = 0;
            set<string>uniq_words;
            set<string>total_uniq_words;
            std::map<string, std::map<string, double>> probability1;
            
            
            if ((argc == 4 && strcmp(argv[2], "--debug")) || (argc == 3)) {
                cout << endl;
                cout << "test data:" << endl;
            }
            
            int performance = 0;
            while (csvin1 >> map_one) {
                ++num_posts_test;
                
                uniq_words = c.unique_words(map_one["content"]);
                
                if ((argc == 4 && strcmp(argv[2], "--debug")) || (argc == 3)) {
                    cout << "  " << "correct = " << map_one["tag"] << ",";
                    //calculates the performance of the program
                    performance = c.classify_test(label_count, uniq_words, probability, 
                                                  total_numb_posts, word_count, map_one, performance);
                    cout << "  " << "content = " << map_one["content"] << endl << endl;
                }
                
            }
            
            //prints out the performance output 
            if ((argc == 4 && strcmp(argv[2], "--debug")) || (argc == 3)) {
                cout << "performance: " << performance << " / " << num_posts_test << " posts predicted correctly" << endl;
            }
        }
        
        catch (csvstream_exception &e) {
            cout << "Error opening file: " << filename2 << endl;
            exit(EXIT_FAILURE);
        }
    }
    else {
        std::cout << "Usage: main TRAIN_FILE TEST_FILE [--debug]" << std::endl;
        exit(EXIT_FAILURE);
    }
    
}

