//
//  main.cpp
//  Project4
//
//  Created by Norin Lavaee on 12/1/17.
//  Copyright © 2017 Norin Lavaee. All rights reserved.


#include <iostream>
#include <vector>
#include <algorithm>
#include <getopt.h>
#include <string>
#include <iomanip>
#include <cstdlib>
#include <limits>
#include <cmath>
#include <deque>


void read_in(std::string &x, std::string &y, std::vector<std::pair<int, int> > & coordinates, std::string &num_entries){
    
    
    int number_entries = std::stoi(num_entries);
    
    int i = 0;
    
    while(number_entries != 0){
        
        std::cin >> x >> y;
        
        int x_coord = std::stoi(x);
        int y_coord = std::stoi(y);
        
        
        coordinates[i] = (std::make_pair(x_coord, y_coord));
        
        --number_entries;
        ++i;
    }
    
}


class DistanceMatrix{
    
public:
    
    double euclidean_distance_nowild(int &x1, int &y1, int &x2, int &y2){
        long double x_plus_y = static_cast<long double>(x1 - x2)*(x1 - x2) + static_cast<long double>(y1 - y2)*(y1 - y2);
        
        return static_cast<double>(std::sqrt(x_plus_y));
    }
    
    void distance_matrix(std::vector<std::pair<int, int> > &coordinates, int number_entries){
        
        distance_mat.resize(number_entries, std::vector<double> (number_entries));
        
        
        for(unsigned int i = 0; i < coordinates.size(); ++i){
            for(unsigned int j = i; j < coordinates.size(); ++j){
                
                distance_mat[i][j] = euclidean_distance_nowild(coordinates[i].first, coordinates[i].second, coordinates[j].first, coordinates[j].second);
                distance_mat[j][i] = euclidean_distance_nowild(coordinates[i].first, coordinates[i].second, coordinates[j].first, coordinates[j].second);
            }
        }
    }
    
    double give_distance(int a, int b){
        return distance_mat[a][b];
    }
    
private:
    std::vector<std::vector<double> > distance_mat;
};


class Prims{
    
public:
    
    double euclidean_distance(int &x1, int &y1, int &x2, int &y2){
    
        if((x1 == 0 && y1 <= 0) || (y1 == 0 && x1 <=0 ) || (x2 == 0 && y2 <= 0) || (y2 == 0 && x2 <=0 )){

        }
        else if(x1 < 0 && y1 < 0 && y2 < 0 && x2 < 0){
            
        }
        
        else if( (x1 < 0 && y1 < 0) || (x2 < 0 && y2 <0)){
            return -1;
        }
        
        
        
        long double x_plus_y = static_cast<long double>(x1 - x2)*(x1 - x2) + static_cast<long double>(y1 - y2)*(y1 - y2);
        
        return static_cast<double>(std::sqrt(x_plus_y));
    }
    
    double euclidean_distance_nowild(int &x1, int &y1, int &x2, int &y2){
        long double x_plus_y = static_cast<long double>(x1 - x2)*(x1 - x2) + static_cast<long double>(y1 - y2)*(y1 - y2);
        
        return static_cast<double>(std::sqrt(x_plus_y));
    }
    
    void MST_algorithm(std::vector<Prims> &MST, std::vector<std::pair<int, int> > &coordinates){
        
        
        int next_index = 0;
        double current_distance = 0;
        MST[0].distance = 0;
        MST[0].preceding_vertex = next_index;
        MST[0].visited = true;
        
        for(unsigned int i = 1; i < coordinates.size(); ++i){
            
            
            for(unsigned int j = 0; j < coordinates.size(); ++j){
                
                if(MST[j].visited == false){
                    
                    
                    current_distance = euclidean_distance(coordinates[next_index].first, coordinates[next_index].second, coordinates[j].first, coordinates[j].second);
                    
                    if(current_distance != -1 && (MST[j].distance > current_distance)){
                        
                        
                        MST[j].distance = current_distance;
                        MST[j].preceding_vertex = next_index;
                        
                        
                    }
                }
            }
            int minimum_index = -1;
            double current_min = std::numeric_limits<double>::infinity();
            
            for(unsigned int k = 1; k < MST.size(); ++k){
    
                if(MST[k].distance < current_min && (MST[k].visited == false)){
                    current_min = MST[k].distance;
                    minimum_index = k;
                }
                
                
            }
            
            if (minimum_index == -1){
                std::cerr << "Cannot construct MST" << '\n';
                exit(1);
            }
            
            MST[minimum_index].visited = true;
            next_index = minimum_index;
        }
        
    }
    
    double all_weight(std::vector<Prims> &MST){
        double weight = 0;
        
        for(unsigned int i = 0; i < MST.size(); ++i){
            weight += MST[i].distance;
        }
        return weight;
    }
    
    
    void print_MST(std::vector<Prims> &MST, double &weight){
        std::cout << weight << '\n';
        for(unsigned int j = 1; j < MST.size(); ++j){
            if(static_cast<int>(j) < MST[j].preceding_vertex){
                std::cout << j << ' ' << MST[j].preceding_vertex << '\n';
            }
            else{
                std::cout << MST[j].preceding_vertex << ' ' << j << '\n';
            }
        }
        std::cout << '\n';
        
    }
    
private:
    
    double distance = std::numeric_limits<double>::infinity();
    
    int preceding_vertex = -1;
    
    bool visited = false;
    
};


class Arbritrary_Insertion{
public:
    
    
    double euclidean_distance_nowild(int &x1, int &y1, int &x2, int &y2){
        long double x_plus_y = static_cast<long double>(x1 - x2)*(x1 - x2) + static_cast<long double>(y1 - y2)*(y1 - y2);
        
        return static_cast<double>(std::sqrt(x_plus_y));
    }
    
    void arbritrary_algo(std::vector<std::pair<int, int> > &coordinates){
        
        double current_distance = 0;
        subtour.push_back(0);
        visited_vertices = std::vector<bool>(coordinates.size(), false);
        visited_vertices[0] = true;
        
        double max = std::numeric_limits<double>::infinity();
        int index_min = 0;

        for(unsigned int h = 1; h < coordinates.size(); ++h){
            
            current_distance = euclidean_distance_nowild(coordinates[0].first, coordinates[0].second, coordinates[h].first, coordinates[h].second);
            
            if(current_distance < max){
                max = current_distance;
                index_min = h;
            }
            
        }
        visited_vertices[index_min] = true;
        subtour.push_back(index_min);
        subtour.push_back(0);
        
        
        
        max = std::numeric_limits<double>::infinity();
        
        for(unsigned int k = 1; k < coordinates.size(); ++k){
            
            if(visited_vertices[k] != true){
                
                for(unsigned j = 0; j < subtour.size() - 1 ; ++j){
                    
                    current_distance = euclidean_distance_nowild(coordinates[subtour[j]].first, coordinates[subtour[j]].second, coordinates[k].first, coordinates[k].second) + euclidean_distance_nowild(coordinates[subtour[j + 1]].first, coordinates[subtour[j + 1]].second, coordinates[k].first, coordinates[k].second) - euclidean_distance_nowild(coordinates[subtour[j]].first, coordinates[subtour[j]].second, coordinates[subtour[j + 1]].first, coordinates[subtour[j+1]].second);
                    
                    if(current_distance < max){
                        max = current_distance;
                        index_min = j;
                    }
                    
                    
                }
                
                current_distance = 0;
                max = std::numeric_limits<double>::infinity();
                
                
                auto it = subtour.begin() + index_min + 1;
                
                subtour.insert(it, k);
                
                visited_vertices[index_min] = true;
            }
        }
        
    }
    
    double calc_weight(std::vector<std::pair<int, int> > &coordinates){
        double weight_for_this = 0;
        
        for(unsigned int i = 0; i < subtour.size() - 1; ++i){
            weight_for_this += euclidean_distance_nowild(coordinates[subtour[i]].first, coordinates[subtour[i]].second, coordinates[subtour[i + 1]].first, coordinates[subtour[i+1]].second);
        }
        return weight_for_this;
    }
    
    
    double print_arbitrary(std::vector<std::pair<int, int> > &coordinates){
        
        double weight_for_this = calc_weight(coordinates);
        
        std::cout << weight_for_this << '\n';
        
        std::cout << subtour[0];
        
        for(unsigned int i = 1; i < subtour.size() - 1; ++i){
            std::cout << " " << subtour[i] ;
            
            
        }
        std::cout << '\n';
        return weight_for_this;
    }
    
    
    
private:
    
    std::vector<int> subtour;
    std::vector<bool> visited_vertices;
    
};

struct Prims_OPTTSP{
    
    double distance = std::numeric_limits<double>::infinity();
    
    int preceding_vertex = -1;
    
    bool visited = false;
};



double MST_for_OPTTSP(std::deque<int> &unused, DistanceMatrix &d){
    
    std::vector<Prims_OPTTSP> MST(unused.size());
    
    double distance_MST = 0;
    int next_index = 0;
    double current_distance = 0;
    
    MST[0].distance = 0;
    MST[0].preceding_vertex = next_index;
    MST[0].visited = true;
    
    
    for(unsigned int i = 1; i < unused.size(); ++i){
        
        
        for(unsigned int j = 0; j < unused.size(); ++j){
            
            if(MST[j].visited == false){
                
                current_distance = d.give_distance(unused[next_index], unused[j]);
                
                if((MST[j].distance > current_distance)){
                    
                    
                    MST[j].distance = current_distance;
                    MST[j].preceding_vertex = next_index;
                    
                    
                }
                
            }
            
            
        }
        
        int minimum_index = -1;
        double current_min = std::numeric_limits<double>::infinity();
        
        for(unsigned int k = 1; k < MST.size(); ++k){
            
            
            if(MST[k].distance < current_min && (MST[k].visited == false)){
                current_min = MST[k].distance;
                minimum_index = k;
            }
            
            
        }
        
        
        if (minimum_index == -1){
            return -1;
        }
        
        
        MST[minimum_index].visited = true;
        next_index = minimum_index;
        
        
        distance_MST += current_min;
    }
    
    return distance_MST;
}


class OPTTSP{
    
public:
    
    
    void initializer(int &number_entries){
        for(int i = 1; i < number_entries; ++i){ //1 2 3 4
            unused.push_back(i);
        }
        
        path.push_back(0);
        
    }
    
    double euclidean_distance_nowild(int &x1, int &y1, int &x2, int &y2){
        long double x_plus_y = static_cast<long double>(x1 - x2)*(x1 - x2) + static_cast<long double>(y1 - y2)*(y1 - y2);
        
        return static_cast<double>(std::sqrt(x_plus_y));
    }
    
    
    
    void FASTTSP_dist(std::vector<std::pair<int, int> > &coordinates, double &FASTTSP_distance){
        
        Arbritrary_Insertion a;
        
        a.arbritrary_algo(coordinates);
        
        FASTTSP_distance =  a.calc_weight(coordinates);
    }
    
    bool promising(double &FASTTSP_distance, double cur_dist, DistanceMatrix &d){
        
        double MST_distance =  MST_for_OPTTSP(unused, d);
        
        double temp_min1  = std::numeric_limits<double>::infinity();
        
        double temp_min2 = std::numeric_limits<double>::infinity();
        
        double dist_1 = 0;
        
        double dist_2 = 0;
        
        for(unsigned int i = 0; i < unused.size(); ++i){
            dist_1 = d.give_distance(path[0], unused[i]);
            dist_2 = d.give_distance(path.back(), unused[i]);
            
            if(dist_1 < dist_2){
                
                if(temp_min1 > dist_1){
                    temp_min1 = dist_1;
                    
                }
                
                else if(temp_min2 > dist_2){
                    temp_min2 = dist_2;
                }
                
            }
            else{
                
                
                if(temp_min2 > dist_2){
                    temp_min2 = dist_2;
                    
                }
                
                else if(temp_min1 > dist_1){
                    temp_min1 = dist_1;
                }


                
            }
            
        }
        
        MST_distance = MST_distance + temp_min2 + temp_min1;
        
        if(cur_dist + MST_distance <= FASTTSP_distance){
            return true;
        }
        else{
            return false;
        }
        
        
        
    }
    
    void gen_perms(std::vector<std::pair<int, int> > &coordinates, double &FASTTSP_distance, double cur_dist, DistanceMatrix &d) {
        
        if(unused.empty()){
            double lel = euclidean_distance_nowild(coordinates[path[0]].first, coordinates[path[0]].second, coordinates[path.back()].first, coordinates[path.back()].second);
            
            cur_dist += lel;
            if(cur_dist <= FASTTSP_distance){
                FASTTSP_distance = cur_dist;
                correct_path = path;
            }
            return;
            
        }
        if (unused.size() >= 4 && !promising(FASTTSP_distance, cur_dist, d)){
            
            return;
        }
        for (unsigned k = 0; k != unused.size(); k++) {
            
            path.push_back(unused.front());
            
            unused.pop_front();
            
            gen_perms(coordinates, FASTTSP_distance, cur_dist + euclidean_distance_nowild(coordinates[path[path.size()-1]].first, coordinates[path[path.size()-1]].second, coordinates[path[path.size()-2]].first, coordinates[path[path.size()-2]].second), d);
            
            unused.push_back(path.back());
            
            path.pop_back();
        }
        
    }
    
    void print_func(std::vector<std::pair<int, int> > &coordinates){
        
        double weight_for_this = 0;
        
        correct_path.push_back(0);
        
        for(unsigned int i = 0; i < correct_path.size()-1; ++i){
            weight_for_this += euclidean_distance_nowild(coordinates[correct_path[i]].first, coordinates[correct_path[i]].second, coordinates[correct_path[i + 1]].first, coordinates[correct_path[i+1]].second);
        }
        
        
        std::cout << weight_for_this << '\n';

        std::cout << correct_path[0];
        
        for(unsigned int i = 1; i < correct_path.size() - 1; ++i){
            std::cout << " " << correct_path[i] ;
            
            
        }
        std::cout << '\n';
    }
private:
    std::deque<int> unused;
    std::vector<int> path;
    std::vector<int> correct_path;
    double path_length = 0;
    
    
};
int main(int argc, char * argv[]) {
    
    std::ios_base::sync_with_stdio(false);
    
    
#ifdef __APPLE__
    if (getenv("STDIN")) freopen(getenv("STDIN"), "r", stdin);
    if (getenv("STDOUT")) freopen(getenv("STDOUT"), "w", stdout);
    if (getenv("STDERR")) freopen(getenv("STDERR"), "w", stderr);
#endif
    
    std::cout << std::setprecision(2);
    std::cout << std::fixed;
    
    Prims p;
    Arbritrary_Insertion a;
    OPTTSP f;
    
    std::vector<std::pair<int, int> > coordinates;
    std::vector<Prims> MST;
    
    std::string x;
    std::string y;
    std::string num_entries;
    
    std::cin >> num_entries;

    int number_entries = std::stoi(num_entries);
    
    
    coordinates = std::vector<std::pair<int, int> >(number_entries);
    
    read_in(x, y, coordinates, num_entries);
    
    MST = std::vector<Prims>(number_entries);
    
    std::deque<int>unused;
    
    std::vector<int>correct_path(number_entries);
    
    
    double FASTTSP_distance = 0;
    
    int gotopt;
    int option_index = 0;
    option long_opts[] = {
        {"mode" , required_argument, 0, 'm'},
        {"help", no_argument, 0, 'h'},
        { nullptr, 0, nullptr, '\0'}
    };
    
    while((gotopt = getopt_long(argc, argv, "hm:", long_opts, &option_index)) != -1){
        
        switch (gotopt) {
                
                
            case 'm':{
                
                
                if(static_cast<std::string>(optarg) == "MST"){
                    
                    DistanceMatrix d;
                    
                    p.MST_algorithm(MST, coordinates);
                    
                    double weight = p.all_weight(MST);
                    
                    p.print_MST(MST, weight);
                    
                }
                else if(static_cast<std::string>(optarg) == "FASTTSP"){
                    
                    a.arbritrary_algo(coordinates);
                    
                    a.print_arbitrary(coordinates);
                    
                }
                else if(static_cast<std::string>(optarg) == "OPTTSP"){
                    
                    double cur_dist = 0;
                    
                    DistanceMatrix d;
                    
                    d.distance_matrix(coordinates, number_entries);
                    
                    f.initializer(number_entries);
                    
                    f.FASTTSP_dist(coordinates, FASTTSP_distance);
                    
                    f.gen_perms(coordinates, FASTTSP_distance, cur_dist, d);
                    
                    f.print_func(coordinates);
                    
                }
                else{
                    std::cerr << "There is something wrong with input parameter" << '\n';
                    exit(1);
                }
                
                break;
            }
                
            case 'h': {
                std::cout << "You ain't gettin none here " << '\n';
                exit(0);
                break;
            }
        }
    }
    
}
