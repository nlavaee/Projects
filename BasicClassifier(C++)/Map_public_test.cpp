#include <iostream>
#include <string>
#include "Map.h"
using namespace std;

int main () {
  // A map stores two types, key and value
  Map<string, double> words;

  // One way to use a map is like an array
  words["hello"] = 1;

  // Maps store a std::pair type, which "glues" one key to one value.
  // The CS term is Tuple, a fixed-size heterogeneous container.
  pair<string, double> tuple;
  tuple.first = "world";
  tuple.second = 2;
  words.insert(tuple);

  // Here's the C++11 way to insert a pair
  words.insert({"pi", 3.14159});

  // Iterate over map contents using a C++11 range-for loop
  // This is the equivalent without C++11:
  // for (Map<string, double>::Iterator i=words.begin();
  //      i != words.end(); ++i) {
  for (auto i:words) {
    auto word = i.first; //key
    auto number = i.second; //value
    cout << word << " " << number << "\n";
  }

  // Check if a key is in the map.  find() returns an iterator.
  auto found_it = words.find("pi");
  if (found_it != words.end()) {
    auto word = (*found_it).first; //key
    auto number = (*found_it).second; //value
    cout << "found " << word << " " << number << "\n";
  }

  // When using the [] notation. An element not found is automatically created.
  // If the value type of the map is numeric, it will always be 0 "by default".
  cout << "bleh " << words["bleh"] << endl;

  cout << "Check that your output matches the .out.correct file." << endl;
}
