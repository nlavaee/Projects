//
//  market.cpp
//  Project2
//
//  Created by Norin Lavaee on 10/7/17.
//  Copyright © 2017 Norin Lavaee. All rights reserved.
//



//make the value a long long when adding two values together and then typecast whole quantity and typecast into an integer, typecast after you divide by two --> add the values typecasted as long
#include <iostream>
#include <vector>
#include <algorithm>
#include <getopt.h>
#include <string>
#include <queue>
#include "Eecs281PQ.h"
#include "P2random.h"
#include <functional>
using std::swap;
using std::vector;
using std::string;
using std::priority_queue;



struct stock_information{
    int timestamp = 0;
    string person = "person";
    unsigned int trader_id = 0;
    unsigned int stock_num = 0;
    int price = 0;
    mutable int quantity = 0;
    unsigned int order = 0;
    
    
    struct BuyComp{
        bool operator()(stock_information &stock_a, stock_information &stock_b) const {
            if( stock_a.price < stock_b.price){//buyers need to go from max to min
                return true;
            }
            else if (stock_b.price < stock_a.price) {
                return false;
            }
            else /*if(stock_a.price == stock_b.price)*/{
                if(stock_a.order > stock_b.order){
                    return true;
                }
                else{
                    return false;
                }
            }
            // return false;
        }
    };
    
    struct SellComp{
        bool operator()(stock_information &stock_a, stock_information &stock_b) const {
            if(stock_a.price < stock_b.price){//sellers need to go from min to max
                return false;
            }
            else if (stock_b.price < stock_a.price){
                return true;
            }
            else /*if(stock_a.price == stock_b.price)*/{
                if(stock_a.order > stock_b.order){
                    return true;
                }
                else{
                    return false;
                }
            }
            //return false;
        }
    };
    
    
};


struct Median{
    priority_queue<int, vector<int>, std::less<int>> max;
    priority_queue<int, vector<int>, std::greater<int>> min;
    
    int median_calc(int price){
        
        if(max.empty()){
            max.push(price);
        }
        else if(min.empty()){
            min.push(price);
            
            if(max.top() > min.top()){
                int temp = max.top();
                max.pop();
                int temp2 = min.top();
                min.pop();
                max.push(temp2);
                min.push(temp);
            }
        }
        
        else{
            
            if(price < max.top()){
                max.push(price);
            }
            else if(price >= max.top()){
                min.push(price);
            }
            
            if(static_cast<int>((min.size() - max.size())) > 1){
                int top = min.top();
                min.pop();
                max.push(top);
            }
            else if(static_cast<int>((max.size() - min.size())) > 1){
                int top = max.top();
                max.pop();
                min.push(top);
                
            }
            
        }
        if(min.size() == max.size()){
            long long median = static_cast<long long>(min.top()) + static_cast<long long>(max.top());
            return static_cast<int>(median/2);
        }
        else{
            if(min.size() > max.size()){
                return min.top();
            }
            else if(max.size() > min.size()){
                return max.top();
            }
        }
        
        
        //int temp = price;
        return 0;
        
    }
    
};

struct TimeTravel{
    
    //indexed by stock number
    
    
    int current_sell_time = 0;
    int current_buy_time = 0;
    int running_max_buy_price = 0;
    int running_min_sell_price = 0;
    int potential_sell_price = 0;
    int potential_sell_time = 0;
    
    
    void calc_time_traveler(int &timestamp, string &person, int &price){
        //read in the time, the stock that came in, and whether it is buy or sell
        
        if(person == "SELL" && running_min_sell_price == 0){
            current_sell_time = timestamp;
            running_min_sell_price = price;
        }
        if(person == "SELL" && running_max_buy_price == 0){
            if(price < running_min_sell_price){
                running_min_sell_price = price;
                current_sell_time = timestamp;
            }
        }
        else if(person == "BUY" && potential_sell_price == 0 && running_min_sell_price != 0){
            if(running_min_sell_price < price && price > running_max_buy_price){
                running_max_buy_price = price;
                current_buy_time = timestamp;
            }
        }
        else if(person == "SELL" && potential_sell_price == 0){
            if(price < running_min_sell_price){
                potential_sell_price = price;
                potential_sell_time = timestamp;
            }
        }
        else if(person == "SELL" && potential_sell_price != 0){
            if(price < running_min_sell_price && price < potential_sell_price){
                potential_sell_price = price;
                potential_sell_time = timestamp;
            }
        }
        else if(person == "BUY" && running_min_sell_price != 0){
            if((static_cast<int>(price) - static_cast<int>(potential_sell_price)) > (static_cast<int>(running_max_buy_price) - static_cast<int>(running_min_sell_price))){
                running_min_sell_price = potential_sell_price;
                running_max_buy_price = price;
                current_sell_time = potential_sell_time;
                current_buy_time = timestamp;
                potential_sell_price = 0;
                
                
            }
        }
        
        
    }
    
};

struct trading{
    
    int profit = 0;
    priority_queue<stock_information, vector<stock_information>,stock_information::BuyComp> buyer;
    priority_queue<stock_information, vector<stock_information>,stock_information::SellComp> seller;
    Median m;
};

struct Trader_information{
    long long bought = 0;
    long long sold = 0;
    long long net_transfer = 0;
    
};

void print_median(int &timestamp, vector<unsigned int> &median_values){
    
    unsigned long i = 0;
    while(i < median_values.size()){
        if(median_values[i] != 0){
            std::cout << "Median match price of Stock " << i << " at time " << timestamp << " is $" << median_values[i] << '\n';
            ++i;
        }
        else{
            ++i;
        }
    }
}



void read_in(vector<trading>& all_stocks, vector<unsigned int> &median_values, bool &med, bool &verbose, bool &info_trader, bool &time_travelers){
    //this will only work for the TL mode (fix for the PR mode)
    stock_information s;
    trading t;
    Trader_information i;
    
    char trash = '/';
    string garbage = "trash";
    string mode = "mode";
    unsigned int num_traders = 0;
    unsigned int num_stocks = 0;
    int current_timestamp = 0;
    int chosen_price = 0;
    unsigned int orders_processed = 0;
    unsigned int seed = 0;
    unsigned int num_orders = 0;
    unsigned int rate = 0;
    int median = 0;
    //int prev_stock = 0;
    
    Median m;
    TimeTravel tt;
    
    
    
    //discard info you don't need
    getline(std::cin, garbage);
    //read in the info from the file
    std::cin >> garbage >> mode >> garbage >> num_traders >> garbage >> num_stocks;
    
    std::stringstream ss;
    if(mode == "PR"){
        // TODO read rest of stdin
        std::cin >> garbage >> seed >> garbage >> num_orders >> garbage >> rate;
        P2random::PR_init(ss, seed, num_traders, num_stocks, num_orders, rate);
    }
    std::istream & inputStream = (mode == "PR") ? ss : std::cin;
    
    all_stocks = vector<trading>(num_stocks);
    vector<Trader_information> trader_info(num_traders);
    median_values = vector<unsigned int>(num_stocks, 0);
    vector<TimeTravel> timetraveler(num_stocks);
    
    //read in the info for each stock
    std::cout << "Processing orders..." << '\n';
    
    while(inputStream >> s.timestamp >> s.person >> trash >> s.trader_id >> trash >> s.stock_num >> trash >> s.price >> trash >> s.quantity){
        
        
        //reset the size of the vector
        if(s.quantity <= 0){
            std::cerr << "There is a problem with quantity" << '\n';
            exit(1);
        }
        if(s.price <= 0){
            std::cerr << "There is a problem with price" << '\n';
            exit(1);
        }
        
        int integer_timestamp = s.timestamp;
        if(integer_timestamp < 0){
            std::cerr << "There is a problem with timestamp" << '\n';
            exit(1);
            
        }
        int integer_traderid = num_traders;
        if(integer_traderid < 0){
            std::cerr << "There is a problem with num_traders" << '\n';
            exit(1);
        }
        
        int integer_numstocks = num_stocks;
        if(integer_numstocks < 0){
            std::cerr << "There is a problem with num_stocks" << '\n';
            exit(1);
        }
        
        if(0 > static_cast<int>(s.trader_id) || static_cast<int>(s.trader_id) > (static_cast<int>(num_traders) - 1)){
            std::cerr << "your trader ids are out of range" << '\n';
            exit(1);
        }
        if(0 > static_cast<int>(s.stock_num) || static_cast<int>(s.stock_num) > (static_cast<int>(num_stocks) - 1)){
            std::cerr << "your stock number are out of range" << '\n';
            exit(1);
        }
        if(s.timestamp < current_timestamp){
            std::cerr << "Your timestamp needs to increase" << '\n';
            exit(1);
        }
        
        if(time_travelers == true){
            timetraveler[s.stock_num].calc_time_traveler(s.timestamp, s.person, s.price);
        }
        
        
    
        //if(med == true || verbose == true || info_trader == true){

        if(s.timestamp != current_timestamp){
            
            //INSERT into here your median print function
            
            if(chosen_price != 0 && med == true){
                //            std::cout << "Median match price of Stock " << prev_stock << " at time " << current_timestamp << " is $" << all_stocks[s.stock_num].m.median_calc(chosen_price) << '\n';
                print_median(current_timestamp, median_values);
                
            }
            current_timestamp = s.timestamp;
        }
        
        
        s.order += 1;
        if(s.person == "BUY"){
            all_stocks[s.stock_num].buyer.push(s);
            
            if(!all_stocks[s.stock_num].seller.empty()){
                stock_information potential = all_stocks[s.stock_num].seller.top();
                while(potential.stock_num == s.stock_num && potential.price <= s.price && s.person != potential.person){
                    
                    if(potential.quantity >= /*s.quantity*/all_stocks[s.stock_num].buyer.top().quantity){
                        
                        //since the quantity you want to sell is greater than the amount the buyer wants, you buy it and dequeue the buyer, need to reduce the quantity the seller has now
                        
                        //update the information with the remaining number of sellable stocks
                        
                        
                        all_stocks[s.stock_num].seller.top().quantity -= /*s.quantity*/all_stocks[s.stock_num].buyer.top().quantity;
                        //push back the updated value
                        
                        //pop off the buyer stock that's order has now been completed
                        
                        int quantity_purchased = all_stocks[s.stock_num].buyer.top().quantity;
                        
                        all_stocks[s.stock_num].buyer.pop();
                        
                        
                        
                        if(potential.order > s.order){
                            chosen_price = s.price;
                        }
                        else{
                            chosen_price = potential.price;
                        }
                        
                        ++orders_processed;
                        
                        if(verbose == true){
                            
                            std::cout << "Trader " << s.trader_id << " purchased " << quantity_purchased << " shares of Stock " << potential.stock_num << " from Trader " << potential.trader_id << " for $" << chosen_price << "/share" <<
                            '\n';
                        }
                        //median = m.median_calc(chosen_price,s);
                        
                        if(med == true){
                            
                            median = all_stocks[potential.stock_num].m.median_calc(chosen_price);
                            median_values[potential.stock_num] = median;
                            //prev_stock = potential.stock_num;
                            
                        }
                        
                        if(info_trader == true){
                            
                            trader_info[s.trader_id].net_transfer = trader_info[s.trader_id].net_transfer - (static_cast<long long>(quantity_purchased) * static_cast<long long>(chosen_price));
                            trader_info[s.trader_id].bought += static_cast<long long>(quantity_purchased);
                            trader_info[potential.trader_id].net_transfer = trader_info[potential.trader_id].net_transfer + (static_cast<long long>(quantity_purchased) * static_cast<long long>(chosen_price));
                            trader_info[potential.trader_id].sold += static_cast<long long>(quantity_purchased);
                        }
                        break;
                        
                    }
                    else if(potential.quantity <= /*s.quantity*/all_stocks[s.stock_num].buyer.top().quantity){
                        //since the quantity you want to sell is less than the amount the buyer has, you sell it, and dequeue the buyer bc they're done
                        all_stocks[s.stock_num].buyer.top().quantity -= potential.quantity;
                        
                        all_stocks[s.stock_num].seller.pop();
                        //make sure to pop off the current element before moving on and adding the updated version to the priority queue
                        
                        if(potential.quantity != 0){
                            if(potential.order > s.order){
                                chosen_price = s.price;
                            }
                            else{
                                chosen_price = potential.price;
                            }
                            
                            ++orders_processed;
                            
                            if(verbose == true){
                                std::cout << "Trader " << s.trader_id << " purchased " << potential.quantity << " shares of Stock " << potential.stock_num << " from Trader " << potential.trader_id << " for $" << chosen_price << "/share" << '\n';
                            }
                            // median = m.median_calc(chosen_price,s);
                            
                            
                            if(med == true){
                                median = all_stocks[potential.stock_num].m.median_calc(chosen_price);
                                median_values[potential.stock_num] = median;
                                //prev_stock = potential.stock_num;
                            }
                            if(info_trader == true){
                                
                                trader_info[s.trader_id].net_transfer = trader_info[s.trader_id].net_transfer - (static_cast<long long>(potential.quantity) * static_cast<long long>(chosen_price));
                                trader_info[s.trader_id].bought += static_cast<long long>(potential.quantity);
                                trader_info[potential.trader_id].net_transfer = trader_info[potential.trader_id].net_transfer + (static_cast<long long>(potential.quantity) * static_cast<int>(chosen_price));
                                trader_info[potential.trader_id].sold += static_cast<long long>(potential.quantity);
                            }
                            
                            
                            if(!all_stocks[s.stock_num].seller.empty()){
                                potential = all_stocks[s.stock_num].seller.top();
                            }
                            else{
                                break;
                            }
                            
                        }
                        else{
                            if(!all_stocks[s.stock_num].seller.empty()){
                                potential = all_stocks[s.stock_num].seller.top();
                            }
                            else{
                                break;
                            }
                            
                        }
                    }
                    
                }
            }
            
        }
        else if(s.person == "SELL"){
            all_stocks[s.stock_num].seller.push(s);
            if(!all_stocks[s.stock_num].buyer.empty()){
                stock_information potential = all_stocks[s.stock_num].buyer.top();
                while(potential.stock_num == s.stock_num && potential.price >= s.price && s.person != potential.person){
                    if(potential.quantity >= /*s.quantity*/ all_stocks[s.stock_num].seller.top().quantity){
                        //since the quantity you want to buy is greater than the amount the seller has, you buy it and dequeue the seller, need to reduce the quantity the buyer wants now
                        
                        all_stocks[s.stock_num].buyer.top().quantity -= /*s.quantity*/ all_stocks[s.stock_num].seller.top().quantity;
                        
                        //all_stocks[s.stock_num].buyer.push(update_buyer);
                        
                        int quantity_purchased = all_stocks[s.stock_num].seller.top().quantity;
                        
                        all_stocks[s.stock_num].seller.pop();
                        
                        
                        if(potential.order > s.order){
                            chosen_price = s.price;
                        }
                        else{
                            chosen_price = potential.price;
                        }
                        
                        ++orders_processed;
                        if(verbose == true){
                            std::cout << "Trader " << potential.trader_id << " purchased " << quantity_purchased  << " shares of Stock " << potential.stock_num << " from Trader " << s.trader_id << " for $" << chosen_price << "/share" << '\n';
                        }
                        
                        if(med == true){
                            median = all_stocks[potential.stock_num].m.median_calc(chosen_price);
                            median_values[potential.stock_num] = median;
                            //prev_stock = potential.stock_num;
                        }
                        if(info_trader == true){
                            
                            trader_info[potential.trader_id].net_transfer = trader_info[potential.trader_id].net_transfer - (static_cast<long long>(quantity_purchased) * static_cast<long long>(chosen_price));
                            trader_info[potential.trader_id].bought += static_cast<long long>(quantity_purchased);
                            trader_info[s.trader_id].net_transfer = trader_info[s.trader_id].net_transfer + (static_cast<long long>(quantity_purchased) * static_cast<long long>(chosen_price));
                            trader_info[s.trader_id].sold += static_cast<long long>(quantity_purchased);
                        }
                        
                        //median = m.median_calc(chosen_price, s);
                        
                        
                        break;
                        
                        
                        
                    }
                    else if(potential.quantity <= /*s.quantity*/ all_stocks[s.stock_num].seller.top().quantity){
                        //since the quantity you want to buy is less than the amount the seller has, you buy it, and dequeue the buyer bc their done
                        
                        all_stocks[s.stock_num].seller.top().quantity -= potential.quantity;
                        
                        all_stocks[s.stock_num].buyer.pop();
                        //make sure to pop off the current top element before sending it back into priority queue
                        
                        
                        if(potential.quantity != 0){
                            if(potential.order > s.order){
                                chosen_price = s.price;
                            }
                            else{
                                chosen_price = potential.price;
                            }
                            
                            ++orders_processed;
                            
                            if(verbose == true){
                                std::cout << "Trader " << potential.trader_id << " purchased " << potential.quantity << " shares of Stock " << potential.stock_num << " from Trader " << s.trader_id << " for $" << chosen_price << "/share" << '\n';
                            }
                            //median = m.median_calc(chosen_price, s);
                            
                            if(med == true){
                                median = all_stocks[potential.stock_num].m.median_calc(chosen_price);
                                median_values[potential.stock_num] = median;
                                //prev_stock = potential.stock_num;
                            }
                            if(info_trader == true){
                                
                                trader_info[potential.trader_id].net_transfer = trader_info[potential.trader_id].net_transfer - (static_cast<long long>(potential.quantity) * static_cast<long long>(chosen_price));
                                trader_info[potential.trader_id].bought += static_cast<long long>(potential.quantity);
                                trader_info[s.trader_id].net_transfer = trader_info[s.trader_id].net_transfer + (static_cast<long long>(potential.quantity) * static_cast<long long>(chosen_price));
                                trader_info[s.trader_id].sold += static_cast<long long>(potential.quantity);
                            }
                            
                            
                            if(!all_stocks[s.stock_num].buyer.empty()){
                                potential = all_stocks[s.stock_num].buyer.top();
                            }
                            else{
                                break;
                            }
                        }
                        else{
                            if(!all_stocks[s.stock_num].buyer.empty()){
                                potential = all_stocks[s.stock_num].buyer.top();
                            }
                            else{
                                break;
                            }
                        }
                    }
                }
            }
            
        }
        
        
    }
    if(med == true){
        print_median(s.timestamp, median_values);
    }
    
//deleted bracket
    std::cout << "---End of Day---" << '\n';
    std::cout << "Orders Processed: " << orders_processed << '\n';
  
    if(info_trader == true){
        std::cout << "---Trader Info---" << '\n';
        for(int i = 0; i < static_cast<int>(trader_info.size()); ++i){
            std::cout << "Trader " << i << " bought " << trader_info[i].bought << " and sold " << trader_info[i].sold << " for a net transfer of $" << trader_info[i].net_transfer << '\n';
        }
    }
    if(time_travelers == true){
        
        
        std::cout << "---Time Travelers---" << '\n';
        for(unsigned int i = 0; i < timetraveler.size(); ++i){
            if(static_cast<int>(timetraveler[i].running_max_buy_price) != 0 && static_cast<int>(timetraveler[i].running_min_sell_price) != 0){
                std::cout << "A time traveler would buy shares of Stock " << i << " at time: " << timetraveler[i].current_sell_time << " and sell these shares at time: " << timetraveler[i].current_buy_time << '\n';
            }
            else{
                std::cout << "A time traveler would buy shares of Stock " << i << " at time: -1 and sell these shares at time: -1" << '\n';
            }
        }
    }
    
}



int main(int argc, char * argv[]){
    
    std::ios_base::sync_with_stdio(false);
    
    
#ifdef __APPLE__
    if (getenv("STDIN")) freopen(getenv("STDIN"), "r", stdin);
    if (getenv("STDOUT")) freopen(getenv("STDOUT"), "w", stdout);
    if (getenv("STDERR")) freopen(getenv("STDERR"), "w", stderr);
#endif
    
    trading t;
    vector<trading> all_stocks;
    vector<unsigned int> median_values;
    
    int gotopt;
    int option_index = 0;
    option long_opts[] = {
        {"verbose" , no_argument, 0, 'v'},
        {"median", no_argument, 0, 'm'},
        {"trader_info", no_argument, 0, 'i'},
        {"time_travelers", no_argument, 0, 't'},
        {"help", no_argument, 0, 'h'},
        { nullptr, 0, nullptr, '\0' }
        
        
    };
    
    bool verbose = false;
    bool median = false;
    bool info_trader = false;
    bool time_travelers = false;
    
    while((gotopt = getopt_long(argc, argv, "vmith", long_opts, &option_index)) != -1){
        switch (gotopt) {
                
            case 'v':
                verbose = true;
                break;
                
            case 'm':
                median = true;
                break;
                
            case 'i':
                info_trader = true;
                break;
                
            case 't':
                time_travelers = true;
                break;
                
            default:
                std::cerr << "You messed up " << '\n';
                exit(1);
                break;
        }
        
    }
    
    
    read_in(all_stocks, median_values, median, verbose, info_trader, time_travelers);
    
    
}
