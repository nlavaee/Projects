#ifndef PAIRINGPQ_H
#define PAIRINGPQ_H

#include "Eecs281PQ.h"
#include <deque>
#include <utility>
#include <queue>
#include <cassert>

// A specialized version of the 'priority_queue' ADT implemented as a pairing priority_queue.
template<typename TYPE, typename COMP = std::less<TYPE>>
class PairingPQ : public Eecs281PQ<TYPE, COMP> {
    using base = Eecs281PQ<TYPE, COMP>;

public:
    // Each node within the pairing heap
    class Node {
        public:
            explicit Node(const TYPE &val)
        : elt{ val }, child{ nullptr }, sibling{ nullptr }, parent{ nullptr}
            {}

            // Description: Allows access to the element at that Node's position.
            // Runtime: O(1) - this has been provided for you.
            const TYPE &operator*() const { return elt; }
            const Node *sibling_ptr() const { return sibling; }
            const Node *child_ptr() const { return child; }

            // The following line allows you to access any private data members of this
            // Node class from within the PairingPQ class. (ie: myNode.elt is a legal
            // statement in PairingPQ's add_node() function).
            friend PairingPQ;

        private:
            TYPE elt;
            Node *child;
            Node *sibling;
            Node *parent;
            // TODO: Add one extra pointer (parent or previous) as desired.
    }; // Node


    // Description: Construct an empty priority_queue with an optional comparison functor.
    // Runtime: O(1)
    explicit PairingPQ(COMP comp = COMP()) :
    base{ comp } {
        // TODO: Implement this function.
        size_of_heap = 0;
        root = nullptr;
    } // PairingPQ()


    // Description: Construct a priority_queue out of an iterator range with an optional
    //              comparison functor.
    // Runtime: O(n) where n is number of elements in range.
    // TODO: when you implement this function, uncomment the parameter names.
    template<typename InputIterator>
    PairingPQ(InputIterator start, InputIterator end, COMP comp = COMP()) :
    base{ comp } {
        // TODO: Implement this function.
        size_of_heap = 0;
        root = nullptr;
      
        while(start != end){
            addNode(*start);
            ++start;
           
        }
        
    } // PairingPQ()


    // Description: Copy constructor.
    // Runtime: O(n)
    PairingPQ(const PairingPQ& other) :
        base{ other.compare } {
        // TODO: Implement this function.
            if(other.root == nullptr){
                size_of_heap = 0;
                root = nullptr;
            }
            
            std::deque<Node*>values;
            
            Node* other_root = other.root;
            values.push_back(other_root);
            
            while(!values.empty()){
                
                Node* current = values.front();
                values.pop_front();
                if(current->sibling != nullptr){
                    values.push_back(current->sibling);
                    
                }
                if(current->child != nullptr){
                    values.push_back(current->child);
                    
                }
                addNode(current->elt);
            }

            //root = new Node (values[0]->elt);
            //root = other.root;
            //size_of_heap = other.size_of_heap;
            
//            for(unsigned int i = 1; i < values.size(); ++i){
//                addNode(values[i]->elt);
//            }
            
           
            
    } // PairingPQ()


    // Description: Copy assignment operator.
    // Runtime: O(n)
    // TODO: when you implement this function, uncomment the parameter names.
    PairingPQ& operator=(const PairingPQ& rhs) {
        // TODO: Implement this function.
        PairingPQ dummy(rhs);
        
        std::swap(root, dummy.root);
        std::swap(size_of_heap, dummy.size_of_heap);

        return *this;
    } // operator=()


    // Description: Destructor
    // Runtime: O(n)
    ~PairingPQ() {
        // TODO: Implement this function.
        //int counter = 0;
        //same code in copy constructor except change the delete current line
        std::deque<Node*>nodes;
        
        if(root == nullptr){
            return;
        }
        
        nodes.push_back(root);
        
        while(!nodes.empty()){
            
            Node* current = nodes.front();
            nodes.pop_front();
            
            if(current->sibling != nullptr){
                
                nodes.push_back(current->sibling);
                //current->sibling = nullptr;
            }
            if(current->child != nullptr){
              
                nodes.push_back(current->child);
                 //current->child = nullptr;
                
            }
            //++counter;
            delete current;
        }
        
       
    } // ~PairingPQ()


    // Description: Assumes that all elements inside the priority_queue are out of order and
    //              'rebuilds' the priority_queue by fixing the priority_queue invariant.
    // Runtime: O(n)
    virtual void updatePriorities() {
        // TODO: Implement this function.
        std::deque<Node*> values;
        unsigned int counter = 0;
        //unsigned int counter2 = 0;
        values.push_back(root);
        while(counter != size_of_heap){
            Node* current = values[counter];
            //values.pop_front();
            if(current->sibling != nullptr){
                values.push_back(current->sibling);
               
                
            }
            if(current->child != nullptr){
                values.push_back(current->child);
                
            }
           
            values[counter]->parent = nullptr;
            values[counter]->sibling = nullptr;
            values[counter]->child = nullptr;
            ++counter;
        }
      
//        while(counter2 != size_of_heap){
//            //clearing all the connections between to the values
//            values[counter2]->parent = nullptr;
//            values[counter2]->sibling = nullptr;
//            values[counter2]->child = nullptr;
//           
//            ++counter2;
//        }
        
        while(values.size() > 1){
            Node* a = values.front();
            values.pop_front();
            Node* b = values.front();
            values.pop_front();
            values.push_back(meld(a, b));
        }
        root = values.front();
        values.pop_front();

    } // updatePriorities()


    // Description: Add a new element to the priority_queue. This has been provided for you,
    //              in that you should implement push functionality in the
    //              addNode function.
    // Runtime: Amortized O(1)
    // TODO: when you implement this function, uncomment the parameter names.
    virtual void push(const TYPE & val) {
        // TODO: Implement this function.
        addNode(val);
        
    } // push()


    // Description: Remove the most extreme (defined by 'compare') element from
    //              the priority_queue.
    // Note: We will not run tests on your code that would require it to pop an
    // element when the priority_queue is empty. Though you are welcome to if you are
    // familiar with them, you do not need to use exceptions in this project.
    // Runtime: Amortized O(log(n))
    virtual void pop() {
        // TODO: Implement this function.
        std::queue<Node*> values;
        if(root->child == nullptr){
            delete root;
            root = nullptr;
            --size_of_heap;
            return;
        }
        
        Node* root_kid = root->child;
        delete root;
        --size_of_heap;
        while(root_kid->sibling != nullptr){
            Node* next_kid = root_kid->sibling;
            root_kid->sibling = nullptr;
            values.push(root_kid);
            root_kid = next_kid;
        }
        values.push(root_kid);
        while(values.size() > 1){
            Node* a = values.front();
            values.pop();
            Node* b = values.front();
            values.pop();
            values.push(meld(a, b));
        }
        root = values.front();
        values.pop();
        
    } // pop()

    
    // Description: Return the most extreme (defined by 'compare') element of
    //              the priority_queue.
    // Runtime: O(1)
    virtual const TYPE & top() const {
        // TODO: Implement this function
        return root->elt;
        // These lines are present only so that this provided file compiles.
        //static TYPE temp; // TODO: Delete this line
        //return temp;      // TODO: Delete or change this line
    } // top()


    // Description: Get the number of elements in the priority_queue.
    // Runtime: O(1)
    virtual std::size_t size() const {
        // TODO: Implement this function
        return size_of_heap; // TODO: Delete or change this line
    } // size()

    // Description: Return true if the priority_queue is empty.
    // Runtime: O(1)
    virtual bool empty() const {
        // TODO: Implement this function
        return size_of_heap == 0; // TODO: Delete or change this line
    } // empty()


    // Description: Updates the priority of an element already in the priority_queue by
    //              replacing the element refered to by the Node with new_value.
    //              Must maintain priority_queue invariants.
    //
    // PRECONDITION: The new priority, given by 'new_value' must be more extreme
    //               (as defined by comp) than the old priority.
    //
    // Runtime: As discussed in reading material.
    // TODO: when you implement this function, uncomment the parameter names.
    void updateElt(Node* node, const TYPE & new_value) {
        // TODO: Implement this function
        node->elt = new_value;
        
        if(node->parent != nullptr){
        Node* guardian = node->parent;
        Node* kin = guardian->child;
        
        if(guardian->child == node){
            guardian->child = node->sibling;
        }
        else{
            while(kin->sibling != node){
                kin = kin->sibling;
            }
            kin->sibling = node->sibling;
            
        }
        //updatePriorities();
            node->sibling = nullptr;
            node->parent = nullptr;
         root = meld(node, root);
        return;
    } // updateElt()
}

    // Description: Add a new element to the priority_queue. Returns a Node* corresponding
    //              to the newly added element.
    // Runtime: Amortized O(1)
    // TODO: when you implement this function, uncomment the parameter names.
    Node* addNode(const TYPE & val) {
        // TODO: Implement this function
       // return nullptr;
        
        if(root == nullptr){
        ++size_of_heap;
        return root = new Node (val);
        }
        else{
        Node* a = new Node (val);
        ++size_of_heap;
        root = meld(root, a);
        if(this->compare(a->elt, root->elt)){
            
            return root->child;
            
        }
        else{
            return root;
        }
        
        //data.push_back(a);
    
        //delete a;
        
    } // addNode()
 }

private:
    // TODO: Add any additional member functions or data you require here.
    // TODO: We recommend creating a 'meld' function (see the Pairing Heap papers).
    std::size_t size_of_heap = 0;
    Node* root = nullptr;
    std::deque<Node*>heap;
    
    Node* meld(Node *a, Node*b){
        assert(a->sibling == nullptr && b->sibling == nullptr);
        if(a == nullptr){
            return b;
        }
        else if(b == nullptr){
            return a;
        }
        if(this->compare(b->elt, a->elt)){
            b->sibling = a->child;
            b->parent = a;
            a->child = b;
            return a;
        }
        else{
            a->sibling = b->child;
            a->parent = b;
            b->child = a;
            return b;
        }
    }
    

};


#endif // PAIRINGPQ_H
