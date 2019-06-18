#ifndef BINARYPQ_H
#define BINARYPQ_H


#include <algorithm>
#include "Eecs281PQ.h"

// A specialized version of the 'heap' ADT implemented as a binary heap.
template<typename TYPE, typename COMP = std::less<TYPE>>
class BinaryPQ : public Eecs281PQ<TYPE, COMP> {
    using base = Eecs281PQ<TYPE, COMP>;

public:
    // Description: Construct an empty heap with an optional comparison functor.
    // Runtime: O(1)
    explicit BinaryPQ(COMP comp = COMP()) :
        base{ comp } {
        // TODO: Implement this function.
    } // BinaryPQ


    // Description: Construct a heap out of an iterator range with an optional
    //              comparison functor.
    // Runtime: O(n) where n is number of elements in range.
    // TODO: when you implement this function, uncomment the parameter names.
    template<typename InputIterator>
    BinaryPQ(InputIterator start, InputIterator end, COMP comp = COMP()) :
    base{ comp }, data{start, end} {
        for(int i = static_cast<int>(data.size()) - 1; i >= 0; --i){
            fixdown(i);
        }

        // TODO: Implement this function.
    } // BinaryPQ


    // Description: Destructor doesn't need any code, the data vector will
    //              be destroyed automaticslly.
    virtual ~BinaryPQ() {
    } // ~BinaryPQ()


    // Description: Assumes that all elements inside the heap are out of order and
    //              'rebuilds' the heap by fixing the heap invariant.
    // Runtime: O(n)
    virtual void updatePriorities() {
        // TODO: Implement this function.
        for(int i = static_cast<int>(data.size()) - 1; i >= 0; --i){
            fixdown(i);
        }
    } // updatePriorities()


    // Description: Add a new element to the heap.
    // Runtime: O(log(n))
    // TODO: when you implement this function, uncomment the parameter names.
    virtual void push(const TYPE & val) {
        
        data.push_back(val);
        
        fixup(static_cast<int>(data.size() - 1));
        // TODO: Implement this function.
    } // push()


    // Description: Remove the most extreme (defined by 'compare') element from
    //              the heap.
    // Note: We will not run tests on your code that would require it to pop an
    // element when the heap is empty. Though you are welcome to if you are
    // familiar with them, you do not need to use exceptions in this project.
    // Runtime: O(log(n))
    virtual void pop() {
        // TODO: Implement this function.
        if(data.size() == 1){
            data.pop_back();
        }
        else{
            
        data.front() = data.back();
        data.pop_back();
            
            fixdown(0);
            
        }
        
    } // pop()


    // Description: Return the most extreme (defined by 'compare') element of
    //              the heap.
    // Runtime: O(1)
    virtual const TYPE & top() const {
        // TODO: Implement this function.
        return data.front();
        
        //These lines are present only so that this provided file compiles.
        static TYPE temp; //TODO: Delete this line
        return temp;      //TODO: Delete or change this line
    } // top()


    // Description: Get the number of elements in the heap.
    // Runtime: O(1)
    virtual std::size_t size() const {
        // TODO: Implement this function.  Might be very simple,
        // depending on your implementation.
        return data.size(); // TODO: Delete or change this line
    } // size()


    // Description: Return true if the heap is empty.
    // Runtime: O(1)
    virtual bool empty() const {
        // TODO: Implement this function.  Might be very simple,
        // depending on your implementation.
        return data.empty(); // TODO: Delete or change this line
    } // empty()


private:
    // Note: This vector *must* be used your heap implementation.
    std::vector<TYPE> data;
    //k -1 /2 because indexing by zero
    
    void fixup(int k){
        
        while(k >= 1 && this->compare(data[(k - 1)/2], data[k])){
            std::swap(data[k], data[(k-1)/2]);
            k = (k-1)/2;
        }

    }
    
    void fixdown(int k){
        
        while(2 * k + 1 <= static_cast<int>(data.size())){
            
            int j = 2 * k + 1;
            
            if((j < static_cast<int>(data.size()) - 1) && this->compare(data[j], data[j + 1])){
                ++j;
            }
               if(this->compare(data[j], data[k])){
                   break;
               }
               std::swap(data[k], data[j]);
               k = j;
            
        }
        
    }
    //TODO: Add any additional member functions or data you require here.

}; // BinaryPQ


#endif // BINARYPQ_H
