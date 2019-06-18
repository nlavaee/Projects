/*
 * Compile this test against your .h files to make sure they compile. Note how
 * the eecs281 priority queues can be constructed with the different types. We
 * suggest adding to this file or creating your own test cases to test your
 * priority queue implementations. 
 *
 * Notice that testPairing() tests the range-based constructor but main() and
 * testPriorityQueue() do not.
 *
 * These tests are NOT a complete test of your priority queues!
 */

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "Eecs281PQ.h"
#include "PoormanPQ.h"
#include "SortedPQ.h"
#include "BinaryPQ.h"
#include "PairingPQ.h"

using namespace std;


// Very basic testing.
void testPriorityQueue(Eecs281PQ<int> *pq, const string &pqType) {
    cout << "Testing priority queue: " << pqType << endl;

    cout << "Testing priority queue: " << pqType << endl;
    cout << "Pushing some values" << endl;
    pq->push(3);
    
    pq->push(4);
    cout << "Testing the size" << endl;
    assert(pq->size() == 2);
    cout << "Testing top" << endl;
    cout << "Top: " << pq->top() << endl;
    assert(pq->top() == 4);
    
    cout << "Testing pop" << endl;
    pq->pop();
    assert(pq->size() == 1);
    cout << "Passed 1/3" << endl;
    assert(pq->top() == 3);
    cout << "Passed 2/3" << endl;
    assert(!pq->empty());
    cout << "Passed 3/3" << endl;
    
    cout << "Testing pop again" << endl;
    pq->pop();
    assert(pq->size() == 0);
    cout << "Passed 1/2" << endl;
    assert(pq->empty());
    cout << "Passed 2/2" << endl;
    
    pq->push(4);
    pq->push(67);
    pq->push(67);
    pq->push(7);
    pq->push(5);
    
    //67 67 7 5 4
    assert(pq->top() == 67);
    pq->pop();
    assert(pq->top() == 67);
    pq->pop();
    assert(pq->top() == 7);
    pq->push(78);
    assert(pq->top() == 78);
    pq->pop();
    assert(pq->top() == 7);
    pq->pop();
    assert(pq->top() == 5);
    pq->pop();
    assert(pq->top() == 4);
    pq->push(2);
    
    
    
    cout << "testPriorityQueue() succeeded!" << endl;
   
    
  
} // testPriorityQueue()


// Test the pairing heap's range-based constructor, copy constructor,
// and operator=().  Still not complete, should have more code, test
// addNode(), updateElt(), etc.
void testPairing(vector<int> & vec) {
    cout << "Testing Pairing Heap separately" << endl;
    Eecs281PQ<int> * pq1 = new PairingPQ<int>(vec.begin(), vec.end());
    Eecs281PQ<int> * pq2 = new PairingPQ<int>(*((PairingPQ<int> *)pq1));
    // This line is different just to show two different ways to declare a
    // pairing heap: as an Eecs281PQ and as a PairingPQ. Yay for inheritance!
    PairingPQ<int> * pq3 = new PairingPQ<int>();
    *pq3 = *((PairingPQ<int> *)pq2);

    pq1->push(3);
    pq2->pop();
    assert(pq1->size() == 3);
    assert(!pq1->empty());
    assert(pq1->top() == 3);
    pq2->push(pq3->top());
    assert(pq2->top() == pq3->top());
    
    pq1->push(4);
    pq1->push(67);
    pq1->push(67);
    pq1->push(7);
    pq1->push(5);
    
    //67 67 7 5 4
    assert(pq1->top() == 67);
    pq1->pop();
    assert(pq1->top() == 67);
    pq1->pop();
    assert(pq1->top() == 7);
    pq1->push(78);
    assert(pq1->top() == 78);
    pq1->pop();
    assert(pq1->top() == 7);
    pq1->pop();
    assert(pq1->top() == 5);
    pq1->pop();
    assert(pq1->top() == 4);
    pq1->push(2);
    
   auto test = pq3->addNode(2);
    pq3->updateElt(test, 7);
    pq3->addNode(5);
    pq3->addNode(10);
    pq3->addNode(-1);
  auto test1 = pq3->addNode(1);
    pq3->updateElt(test1, 70);
    
    
    PairingPQ<int> * pq5 = new PairingPQ<int>();
    pq5->push(12);
    pq5->push(17);
    pq5->push(19);
    PairingPQ<int> * pq4 = new PairingPQ<int>(*((PairingPQ<int> *)pq5));
    assert(pq4->top() == 19);
    pq4->pop();
    assert(pq4->top() == 17);
    pq4->pop();
    assert(pq4->top() == 12);
    
    
    
    
    PairingPQ<int> * pq6 = new PairingPQ<int>();
    pq6->push(27);
    pq6->push(88);
    pq6->push(900);
    pq6->push(23);
    PairingPQ<int> * pq7 = new PairingPQ<int>();
    pq7->push(45);
    pq7->push(34);
    pq7->push(33);
    
    pq7 = pq6;
    assert(pq7->size() == 4);
    assert(pq6->size() == 4);
    pq6 = pq7;
    
   

    PairingPQ<int> * pq8 = new PairingPQ<int>();
    pq8->push(27);
    pq8->push(4);
    pq8->push(54);
    pq8->updatePriorities();
    

    cout << "Basic tests done, calling destructors" << endl;

    delete pq1;
    delete pq2;
    delete pq3;
    delete pq4;
    delete pq5;
    delete pq6;
    delete pq8;
    //delete pq7;

    cout << "testPairing() succeeded" << endl;
} // testPairing()


int main() {
    // Basic pointer, allocate a new PQ later based on user choice.
    Eecs281PQ<int> *pq;
    vector<string> types{ "Poorman", "Sorted", "Binary", "Pairing" };
    int choice;

    cout << "PQ tester" << endl << endl;
    for (size_t i = 0; i < types.size(); ++i)
        cout << "  " << i << ") " << types[i] << endl;
    cout << endl;
    cout << "Select one: ";
    cin >> choice;

    if (choice == 0) {
        pq = new PoormanPQ<int>;
    } // if
    else if (choice == 1) {
        pq = new SortedPQ<int>;
    } // else if
    else if (choice == 2) {
       pq = new BinaryPQ<int>;
    } // else if
    else if (choice == 3) {
        pq = new PairingPQ<int>;
    } // else if
    else {
        cout << "Unknown container!" << endl << endl;
        exit(1);
    } // else
   
    testPriorityQueue(pq, types[choice]);

    if (choice == 3) {
        vector<int> vec;
        vec.push_back(0);
        vec.push_back(1);
        testPairing(vec);
    } // if

    // Clean up!
    delete pq;

    return 0;
} // main()
