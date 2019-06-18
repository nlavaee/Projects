#ifndef BINARYPQ_H
#define BINARYPQ_H


#include <algorithm>
#include "Eecs281PQ.h"
using std::swap;
using std::vector;

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
        // TODO: Implement this function.
        
    } // BinaryPQ


    // Description: Destructor doesn't need any code, the data vector will
    //              be destroyed automaticslly.
    virtual ~BinaryPQ() {
    } // ~BinaryPQ()
//pop fix down (CORRECT)

    // Description: Assumes that all elements inside the heap are out of order and
    //              'rebuilds' the heap by fixing the heap invariant.
    // Runtime: O(n)
    virtual void updatePriorities() {
       
        // TODO: Implement this function.
        //FIX UP ELLAINA
        int start_index = static_cast<int>(data.size()/2);
        //int left_child = start_index*2;
        //int right_child = start_index*2 + 1;
        int parent = (start_index - 1)/2;
        
//        while(data[right_child] > data[left_child] && (2*start_index <= data.size())){
//            swap(data[start_index], data[right_child]);
//            start_index = right_child;
//        }
        for(unsigned int i = 0; i < data.size()/2; ++i){
//            
//        while((start_index < static_cast<unsigned long>(data.size())) && this->compare(data[parent], data[start_index])/*data[start_index] > data[parent]*/){
//            
//            unsigned long child = start_index*2;
//            
//            if((start_index < static_cast<unsigned long>(data.size())) && this->compare(data[start_index], static_cast<unsigned long>(data[start_index + 1]))){
//                ++start_index; //go to the right child
//            }
//            if(/*data[start_index] >= data[child]*/this->compare(data[child], data[start_index])){ //checks to see if the parent you have is greater than the child then break
//                break;
//            }
//            //otherwise swap the child and the start_index bc child is greater
//            swap(data[start_index], data[child]);
//            start_index = child;
//        }
            
            
            //unsigned long index = data.size()/2;
            // int left_child = 2*index;
            //int right_child = 2*index + 1;
            //unsigned long parent = (index - 1) / 2; //?? is this correct?
            while(start_index >= 1 && this->compare(data[parent], data[start_index])){
                swap(data[start_index], data[parent]);
                start_index = parent;
            }

            
            
        
    }
        
    } // updatePriorities()


    // Description: Add a new element to the heap.
    // Runtime: O(log(n))
    // TODO: when you implement this function, uncomment the parameter names.  (push fixes up)
    virtual void push(const TYPE & val) {
        // TODO: Implement this function.
//        unsigned long new_elem = ++data.size();
//        data[new_elem] = val;
        //implement fix up --> ellaina
        
        data.push_back(val);
        int index = static_cast<int>(data.size()) - 1;
        // int left_child = 2*index;
        //int right_child = 2*index + 1;
        int parent = (index - 1) / 2; //?? is this correct?
        
        while(index >= 1 && this->compare(data[parent], data[index])){
            swap(data[index], data[parent]);
            index = parent;
        }
        

        
    } // push()


    // Description: Remove the most extreme (defined by 'compare') element from
    //              the heap.
    // Note: We will not run tests on your code that would require it to pop an
    // element when the heap is empty. Though you are welcome to if you are
    // familiar with them, you do not need to use exceptions in this project.
    // Runtime: O(log(n))
    virtual void pop() { //fix_down correct  -->Ellaina
        
        data.front() = data.back();
        
        data.pop_back();
        
        int start_index = 0;
        
        int left_child = 0;
        
        while(2 * (start_index + 1) <= static_cast<int>(data.size())){
             left_child = 2*(start_index + 1);
            if(left_child < static_cast<int>(data.size()) && this->compare(data[left_child], data[left_child + 1])){
                ++left_child;
            }
            if(this->compare(data[start_index], data[left_child])){
                break;
            }
            swap(data[start_index], data[left_child]);
            start_index = left_child;
        }
        
        //int left_child = start_index*2;
        //int right_child = start_index*2 + 1;
        //int parent = (start_index - 1)/2;
        
        //        while(data[right_child] > data[left_child] && (2*start_index <= data.size())){
        //            swap(data[start_index], data[right_child]);
        //            start_index = right_child;
        //        }
//        while((start_index < static_cast<int>(data.size())) && this->compare(data[parent], data[start_index])/*data[start_index] > data[parent]*/){
//            
//            int child = start_index*2;
//            
//            if((start_index < static_cast<int>(data.size())) && this->compare(data[start_index], data[start_index + 1])){
//                ++start_index; //go to the right child
//            }
//            if(/*data[start_index] >= data[child]*/this->compare(data[child], data[start_index])){ //checks to see if the parent you have is greater than the child then break
//                break;
//            }
//            //otherwise swap the child and the start_index bc child is greater
//            swap(data[start_index], data[child]);
//            start_index = child;
//        }
//
    } // pop()


    // Description: Return the most extreme (defined by 'compare') element of
    //              the heap.
    // Runtime: O(1)
    virtual const TYPE & top() const {
        // TODO: Implement this function.
        //updatePriorities();
        return data.front();
        //These lines are present only so that this provided file compiles.
        //static TYPE temp; //TODO: Delete this line
        //return temp;      //TODO: Delete or change this line
    } // top()


    // Description: Get the number of elements in the heap.
    // Runtime: O(1)
    virtual std::size_t size() const {
        // TODO: Implement this function.  Might be very simple,
        // depending on your implementation.
        return data.size();
       // return 0; // TODO: Delete or change this line
    } // size()


    // Description: Return true if the heap is empty.
    // Runtime: O(1)
    virtual bool empty() const {
        // TODO: Implement this function.  Might be very simple,
        // depending on your implementation.
        return data.empty();
        //return true; // TODO: Delete or change this line
    } // empty()


private:
    // Note: This vector *must* be used your heap implementation.
    std::vector<TYPE> data;
    
   
     // getEl()

    //TODO: Add any additional member functions or data you require here.

}; // BinaryPQ


#endif // BINARYPQ_H
