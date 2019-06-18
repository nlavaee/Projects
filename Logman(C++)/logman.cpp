//
//  main.cpp
//  Project03
//
//  Created by Norin Lavaee on 10/31/17.
//  Copyright © 2017 Norin Lavaee. All rights reserved.
//

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string.h>
#include <deque>
#include <unordered_map>

//template<typename F>


//add help

struct MasterLog{
    std::string timestamp = "";
    std::string category = "";
    std::string message = "";
    int entryID = 0;
    long long timestamp_numb = 0;
    //int entryID = 0;
};

//struct VecofTimestamp{
//    long long timestamp = 0; //change to long long
//    int entryID = 0;
//};


void timestamp_search(long long timestamp1, long long timestamp2, std::vector<MasterLog> &master, std::deque<int> &preexcerptlist){
    
    preexcerptlist.clear();
    
    //move lambda in here
    auto masterlog_comp = [&](const MasterLog &loga, const MasterLog &logb){
        return loga.timestamp_numb < logb.timestamp_numb;
    };
    
    MasterLog ts1;
    MasterLog ts2;
    
    ts1.timestamp_numb = timestamp1;
    ts2.timestamp_numb = timestamp2;
    
    int searches_found = 0;
    
    auto it1 = std::lower_bound(master.begin(), master.end(), ts1, masterlog_comp);
    auto it2 = std::upper_bound(master.begin(), master.end(), ts2, masterlog_comp); //because want to get the next higher value
    
    while(it1 < it2){
        preexcerptlist.push_back(static_cast<int>(it1 - master.begin()));
        ++it1;
        ++searches_found;
    }
    std::cout << "Timestamps search: " << searches_found <<  " entries found" << '\n';
}


void print_exlist(std::deque<int> &excerptlist, std::vector<MasterLog> &master){
    
    for(unsigned int i = 0; i < excerptlist.size(); ++i){
        std::cout << i << '|' << master[excerptlist[i]].entryID << '|' << master[excerptlist[i]].timestamp << '|' << master[excerptlist[i]].category << '|' << master[excerptlist[i]].message << '\n';
    }
}

void matching_timestamp(long long timestamp, std::vector<MasterLog> &master, std::deque<int> &preexcerptlist){
    
    preexcerptlist.clear();
    
    auto master_comp = [&](const MasterLog &loga, const MasterLog &logb){
        return loga.timestamp_numb < logb.timestamp_numb;
    };
    
    
    int searches_found = 0;
    
     MasterLog ts;
    
    ts.timestamp_numb = timestamp;
    
    auto it = std::lower_bound(master.begin(), master.end(), ts, master_comp);
    
    while((*it).timestamp_numb == timestamp){
        preexcerptlist.push_back(static_cast<int>(it - master.begin()));
        ++it;
        ++searches_found;
    }
    std::cout << "Timestamp search: " << searches_found <<  " entries found" << '\n';
}

void recent_previous_search(std::deque<int> &excerptlist, std::deque<int> &preexcerptlist){
    
    int num_appended = 0;
    
    for(unsigned int i = 0; i < preexcerptlist.size(); ++i){
        excerptlist.push_back(preexcerptlist[i]);
        ++num_appended;
    }
    std::cout << num_appended << " log entries appended" << '\n';
    
    
}


void g_mostrecent(std::deque<int> &preexcerptlist, std::vector<MasterLog> &master){
    
    for(unsigned int i = 0; i < preexcerptlist.size(); ++i){
        std::cout << master[preexcerptlist[i]].entryID << '|' << master[preexcerptlist[i]].timestamp << '|' << master[preexcerptlist[i]].category << '|' << master[preexcerptlist[i]].message << '\n';
    }
}

void fill_category_searcher(std::vector<MasterLog> &master, std::unordered_map<std::string, std::deque<int> > &category_searcher){
    
    //std::pair <std::string, std::deque<int> > value;
    
    // std::deque<int> values;
    
    for(unsigned int i = 0; i < master.size(); ++i){
        
        //you wanted
        std::string category = master[i].category;
        
        
        //std::tolower(category);
        for(unsigned int j = 0; j < category.size(); ++j){
            category[j] = static_cast<char>(std::tolower(category[j]));
        }
        //
        //        if(category_searcher.find(category) == category_searcher.end()){
        //
        //            values.push_back(i);
        //
        //            category_searcher.insert(std::make_pair(category, values));
        //
        //            //category_searcher.find(category)->second.push_back(i);
        //
        //            values.clear();
        //
        //        }
        //        else{
        //            category_searcher.find(category)->second.push_back(i);
        //        }
        
        category_searcher[category].push_back(i);
    }
}

void fill_keyword_searcher(std::vector<MasterLog> &master, std::unordered_map<std::string, std::deque<int> > &keyword_searcher){
    
    //rewrite by adding category to message and doing as whole
    // std::deque<int> values;
    
    std::deque<std::string> words;
    
    std::string key_word = "";
    
    for(unsigned int i = 0; i < master.size(); ++i){
        
        words.clear();
        
        std::string category = master[i].category;
        std::string message = master[i].message;
        
        for(unsigned int j = 0; j < category.size(); ++j){
            category[j] = static_cast<char>(std::tolower(category[j]));
        }
        for(unsigned int k = 0; k < message.size(); ++k){
            message[k] = static_cast<char>(std::tolower(message[k]));
        }
        
        std::string message_catgeory = category + " " + message;
        
        for(unsigned int h = 0; h < message_catgeory.size(); ++h){
            
            if(std::isalnum(message_catgeory[h])){
                key_word += message_catgeory[h];
            }
            else if(key_word != ""){
                words.push_back(key_word);
                key_word = "";
            }
        }
        if(key_word != ""){
            words.push_back(key_word);
            key_word = "";
        }
        
        
        
        for(unsigned int l = 0; l < words.size(); ++l){
            
            
            //            if(keyword_searcher.find(words[l]) == keyword_searcher.end()){
            //
            //                values.push_back(i);
            //
            //                keyword_searcher.insert(std::make_pair(words[l], values));
            //
            //                values.clear();
            //            }
            //
            //            else{
            //                unsigned long size_of_deque = keyword_searcher.find(words[l])->second.size();
            //                if(keyword_searcher.find(words[l])->second[size_of_deque - 1] != static_cast<int>(i)){
            //                    keyword_searcher.find(words[l])->second.push_back(i);
            //                }
            //
            //            }
            if(keyword_searcher[words[l]].size() == 0 || keyword_searcher[words[l]].back() != static_cast<int>(i)){
                keyword_searcher[words[l]].push_back(i);
            }
            
            
        }
    }
    
    
}

void category_searches(std::unordered_map<std::string, std::deque<int> > &category_searcher, std::deque<int> &preexcerptlist, std::string &user_input){
    
    preexcerptlist.clear();
    
    std::string category_values;
    
   // int num_searches = 0;
    
    for(unsigned int i = 0; i < user_input.size(); ++i){
        
        user_input[i] = static_cast<char>(std::tolower(user_input[i]));
    }
    
    if(/*category_searcher.find(user_input) != category_searcher.end()*/ category_searcher[user_input].size() != 0){
        
       preexcerptlist =  category_searcher[user_input];
        
    }
    
//    for(unsigned int i = 0; i < preexcerptlist.size(); ++i){
//        preexcerptlist[i] = master[preexcerptlist[i]].entryID;
//        ++num_searches;
//    }
    
    
    std::cout << "Category search: " << preexcerptlist.size() << " entries found" << '\n';
    
}

void keyword_searches(std::unordered_map<std::string, std::deque<int> > &keyword_searcher, std::deque<int> &preexcerptlist, std::string &user_input){
    
    preexcerptlist.clear();
    
   // int num_searches = 0;
    
    std::deque<int> values;
    
    
    
    int count = 0;
    
    std::deque<std::string> words;
    
    std::string key_word = "";
    
    for(unsigned int i = 0; i < user_input.size(); ++i){
        
        if(std::isalnum(user_input[i])){
            
            key_word += static_cast<char>(std::tolower(user_input[i]));
        }
        else if(key_word != ""){
            words.push_back(key_word);
            key_word = "";
        }
    }
    if(key_word != ""){
        words.push_back(key_word);
        key_word = "";
    }
    
    
    for(unsigned int i = 0; i < words.size(); ++i){
        
        if(/*keyword_searcher.find(words[i]) != keyword_searcher.end()*/ keyword_searcher[words[i]].size() != 0 && count == 0){
            
            //preexcerptlist = keyword_searcher.find(words[i])->second;
            preexcerptlist = keyword_searcher[words[i]];
            ++count;
            
        }
        else if(/*keyword_searcher.find(words[i]) != keyword_searcher.end()*/keyword_searcher[words[i]].size() != 0 ){
            
            
            //values = keyword_searcher.find(words[i])->second;
            
            values = keyword_searcher[words[i]];
            std::deque<int> intersection(values.size() + preexcerptlist.size());
            auto it = std::set_intersection(values.begin(), values.end(), preexcerptlist.begin(), preexcerptlist.end(), intersection.begin());
            intersection.resize(it-intersection.begin());
            preexcerptlist = intersection;
            
        }
        else{
            preexcerptlist.clear();
            std::cout << "Keyword search: 0 entries found" << '\n';
            return;
        }
    }
    
//    for(unsigned int i = 0; i < preexcerptlist.size(); ++i){
//        preexcerptlist[i] = master[i].entryID;
//        ++num_searches;
//    }
    
    std::cout << "Keyword search: " << preexcerptlist.size() << " entries found" << '\n';
    
}



int main(int argc, char * argv[]) {
    
    std::ios_base::sync_with_stdio(false);
    
#ifdef __APPLE__
    if (getenv("STDIN")) freopen(getenv("STDIN"), "r", stdin);
    if (getenv("STDOUT")) freopen(getenv("STDOUT"), "w", stdout);
    if (getenv("STDERR")) freopen(getenv("STDERR"), "w", stderr);
#endif
    
    MasterLog m;
    //VecofTimestamp v;
    // TimestampComparator t;
    
    
    int entryID = 0;
    
    argc = entryID;
    ++argc;
    
    std::string timestamp_colons;
    
    std::string line;
    
    std::vector<MasterLog> master;
    //std::vector<VecofTimestamp> timestamps;
    std::deque<int> excerptlist;
    std::deque<int> preexcerptlist;
    std::unordered_map<std::string, std::deque<int> > category_searcher;
    std::unordered_map<std::string, std::deque<int> > keyword_searcher;
    
    std::string filename = std::string(argv[1]);
    
    if(filename[0] == '-'){
        std::cout << "I can't help you" << '\n';
        exit(0);
        
    }
    std::ifstream ifs;
    ifs.open(filename);
    
    bool did_search = false;
    
    if(ifs.is_open()){
        
        
        while(getline(ifs, line)){
            
            std::stringstream ss(line);
            
            getline(ss, timestamp_colons, '|');
            
            m.timestamp = timestamp_colons;
            
            timestamp_colons.erase(std::remove(timestamp_colons.begin(), timestamp_colons.end(), ':'), timestamp_colons.end());
            
            m.timestamp_numb = std::stoll(timestamp_colons);
            
            getline(ss, m.category, '|');
            
            getline(ss, m.message);
            
            m.entryID = entryID;
            
            master.push_back(m);
            
            
            
           // timestamps.push_back(v);
            
            ++entryID;
            
            
            
        }
        
        
        std::cout << entryID << " entries read" << '\n';
        
        ifs.close();
        
        master.resize(entryID);
        //timestamps.resize(entryID);
        
        
        auto comp_master = [&master](const MasterLog &loga, const MasterLog &logb){
            
            if(loga.timestamp < logb.timestamp){
                return true;
            }
            else if(loga.timestamp > logb.timestamp){
                return false;
            }
            else{
                
                int string_comparison = strcasecmp(loga.category.c_str(), logb.category.c_str());
                
                if(string_comparison < 0 ){
                    return true;
                }
                else if(string_comparison > 0){
                    return false;
                }
                else{
                    if(loga.entryID < logb.entryID){
                        return true;
                    }
                    else{
                        return false;
                    }
                }
            }
            
        };
        
        
        
//        auto sorting_excerpt = [&master](const int &loga, const int &logb){
//            
//            if(master[loga].timestamp < master[logb].timestamp){
//                return true;
//            }
//            else if(master[loga].timestamp > master[logb].timestamp){
//                return false;
//            }
//            else{
//                
//                int string_comparison = strcasecmp(master[loga].category.c_str(), master[logb].category.c_str());
//                
//                if(string_comparison < 0 ){
//                    return true;
//                }
//                else if(string_comparison > 0){
//                    return false;
//                }
//                else{
//                    if(loga < logb){
//                        return true;
//                    }
//                    else{
//                        return false;
//                    }
//                }
//            }
//            
//        };
//        
        
        
        std::stable_sort(master.begin(), master.end(), comp_master);
        
        //sort the master vector into a vector indices
        
        //
        std::string user_input;
        
        //std::cin >> user_input;
        
        std::cout << "% ";
        
        
        //t m c k a r d b e s l g p #
        
        
        std::string trash;
        
        fill_category_searcher(master, category_searcher);
        fill_keyword_searcher(master, keyword_searcher);
        
        while(getline(std::cin, user_input) && user_input != "q"){
            
            
            switch (user_input[0]) {
                    
                case 's': {
                    
                    if(!excerptlist.empty()){
                        
                        std::cout << "excerpt list sorted" << '\n';
                        std::cout << "previous ordering:" << '\n';
                        std::cout << 0 << '|' << master[excerptlist[0]].entryID << '|' << master[excerptlist[0]].timestamp << '|' << master[excerptlist[0]].category << '|' << master[excerptlist[0]].message << '\n';
                        
                        std::cout << "..." << '\n';
                        
                        std::cout << excerptlist.size() - 1 << '|' << master[excerptlist[excerptlist.size() - 1]].entryID << '|' << master[excerptlist[excerptlist.size() - 1]].timestamp << '|' << master[excerptlist[excerptlist.size() - 1]].category << '|' << master[excerptlist[excerptlist.size() - 1]].message << '\n';
                        
                        std::stable_sort(excerptlist.begin(), excerptlist.end());
                        
                        std::cout << "new ordering:" << '\n';
                        std::cout << 0 << '|' << master[excerptlist[0]].entryID << '|' << master[excerptlist[0]].timestamp << '|' << master[excerptlist[0]].category << '|' << master[excerptlist[0]].message << '\n';
                        
                        std::cout << "..." << '\n';
                        
                        std::cout << excerptlist.size() - 1 << '|' << master[excerptlist[excerptlist.size() - 1]].entryID << '|' << master[excerptlist[excerptlist.size() - 1]].timestamp << '|' << master[excerptlist[excerptlist.size() - 1]].category << '|' << master[excerptlist[excerptlist.size() - 1]].message << '\n';
                        
                    }
                    else{
                        std::cout << "excerpt list sorted" << '\n';
                        std::cout << "(previously empty)" << '\n';
                    }
                    break;
                }
                case 'c': {
                    
                    did_search = true;
                    
                    std::stringstream ss(user_input);
                    
                    getline(ss, user_input, ' ');
                    
                    getline(ss, user_input);
                    
                    category_searches(category_searcher, preexcerptlist, user_input);
                    
                    break;
                }
                    
                case 'k': {
                    
                    did_search = true;
                    
                    std::stringstream ss(user_input);
                    
                    getline(ss, user_input, ' ');
                    
                    getline(ss, user_input);
                    
                    keyword_searches(keyword_searcher, preexcerptlist, user_input);
                    
                    break;
                }
                case 'm': {
                    
                    long long a = 0;
                    
                    std::stringstream ss(user_input);
                    
                    getline(ss, trash, ' ');
                    
                    getline(ss, trash);
                    
                    trash.erase(std::remove(trash.begin(), trash.end(), ':'), trash.end());
                    
                    a = std::stoll(trash);
                    
                    
                    if(trash.size() != 10){
                        
                        std::cerr << "Timestamp1 is the wrong length" << '\n';
                    }
                    else{
                        matching_timestamp(a, master, preexcerptlist);
                        did_search = true;
                    }
                    break;
                }
                case 't':{
                    
                    
                    long long a = 0;
                    
                    long long b = 0;
                    
                    std::stringstream ss(user_input);
                    
                    getline(ss, trash, ' ');
                    
                    getline(ss, trash, '|');
                    
                    if(trash.size() != 14){
                        
                        std::cerr << "Timestamp1 is the wrong length" << '\n';
                    }
                    else{
                        trash.erase(std::remove(trash.begin(), trash.end(), ':'), trash.end());
                        
                        a = std::stoll(trash);
                        
                        getline(ss, trash);
                        
                        
                        
                    }
                    if(trash.size() != 14){
                        
                        std::cerr << "Timestamp2 is the wrong length" << '\n';
                        
                    }
                    else{
                        trash.erase(std::remove(trash.begin(), trash.end(), ':'), trash.end());
                        
                        b = std::stoll(trash);
                        
                        timestamp_search(a, b, master, preexcerptlist);
                        did_search = true;
                    }
                    break;
                }
                case 'a': {
                    
                    int a = 0;
                    
                    std::vector<int> appendingmaster(master.size());
                    
                    std::stringstream ss(user_input);
                    
                    getline(ss, user_input, ' ');
                    
                    getline(ss, user_input);
                    
                    a = std::stoi(user_input);
                    
                    if(a < 0 || a > static_cast<int>(master.size()) - 1){
                        
                        std::cerr << "There is no entry at this index for command a" << '\n';
                    }
                    else{
                        
                        //excerptlist.push_back(a);
                        for(unsigned int i = 0; i < master.size(); ++i){
                            appendingmaster[master[i].entryID] = i;
                        }
                        excerptlist.push_back(appendingmaster[a]);
                        std::cout << "log entry " << a << " appended" << '\n';
                    }
                    
                    break;
                }
                    
                case 'd' : {
                    
                    int a = 0;
                    
                    std::stringstream ss(user_input);
                    
                    getline(ss, user_input, ' ');
                    
                    getline(ss, user_input);
                    
                    a = std::stoi(user_input);
                    
                    if(a < 0 || a > static_cast<int>(excerptlist.size()) - 1){
                        
                        std::cerr << "There is no entry at this index for command a" << '\n';
                    }
                    else{
                        excerptlist.erase(excerptlist.begin() + a);
                        
                        std::cout << "Deleted excerpt list entry " << a << '\n';
                    }
                    break;
                }
                    
                case 'b' : {
                    
                    int a = 0;
                    
                    // int b = 0;
                    
                    std::stringstream ss(user_input);
                    
                    getline(ss, user_input, ' ');
                    
                    getline(ss, user_input);
                    
                    a = std::stoi(user_input);
                    
                    if(a < 0 || a > static_cast<int>(excerptlist.size()) - 1){
                        
                        std::cerr << "There is no entry at this index for command a" << '\n';
                    }
                    
                    else{
                        // excerptlist.push_front(excerptlist[a]);
                        
                        // b = a + 1;
                        
                        // excerptlist.erase(excerptlist.begin() + b);
                        
                        std::rotate(excerptlist.begin(), excerptlist.begin() + a, excerptlist.begin() + a + 1);
                        
                        std::cout << "Moved excerpt list entry " << a << '\n';
                    }
                    break;
                }
                    
                case 'e' : {
                    
                    int a = 0;
                    
                    std::stringstream ss(user_input);
                    
                    getline(ss, user_input, ' ');
                    
                    getline(ss, user_input);
                    
                    a = std::stoi(user_input);
                    
                    if(a < 0 || a > static_cast<int>(excerptlist.size()) - 1){
                        
                        std::cerr << "There is no entry at this index for command a" << '\n';
                    }
                    else{
                        //excerptlist.push_back(excerptlist[a]);
                        
                        //excerptlist.erase(excerptlist.begin() + a);
                        
                        std::rotate(excerptlist.begin() + a, excerptlist.begin() + a + 1, excerptlist.end());
                        
                        
                        std::cout << "Moved excerpt list entry " << a << '\n';
                    }
                    break;
                }
                case 'p' : {
                    
                    print_exlist(excerptlist, master);
                    
                    break;
                }
                    
                case 'r' : {
                    
                    if(did_search ==  false){
                        std::cerr << "No previous search has occured" << '\n';
                    }
                    else{
                        recent_previous_search(excerptlist, preexcerptlist);
                    }
                    break;
                }
                    
                case 'g': {
                    
                    if(did_search == false){
                        std::cerr << "No previous search has occured" << '\n';
                    }
                    else{
                        g_mostrecent(preexcerptlist, master);
                    }
                    break;
                }
                    
                case '#':{
                    
                    break;
                }
                    
                case 'l' : {
                    
                    std::cout << "excerpt list cleared" << '\n';
                    if(!excerptlist.empty()){
                        std::cout << "previous contents:" << '\n';
                        std::cout << 0 << '|' << master[excerptlist[0]].entryID << '|' << master[excerptlist[0]].timestamp << '|' << master[excerptlist[0]].category << '|' << master[excerptlist[0]].message << '\n';
                        
                        std::cout << "..." << '\n';
                        
                        std::cout << excerptlist.size() - 1 << '|' << master[excerptlist[excerptlist.size() - 1]].entryID << '|' << master[excerptlist[excerptlist.size() - 1]].timestamp << '|' << master[excerptlist[excerptlist.size() - 1]].category << '|' << master[excerptlist[excerptlist.size() - 1]].message << '\n';
                        excerptlist.clear();
                    }
                    else{
                        std::cout << "(previously empty)" << '\n';
                    }
                    break;
                }
                    
                default: {
                    
                    std::cerr << "This is not a valid command" << '\n';
                    break;
                }
                    
            }
            std::cout << "% ";
            
        }
        
    }
    
    return 0;
    
}
