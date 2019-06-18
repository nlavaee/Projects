#include "BinarySearchTree.h"
#include <iostream>
#include <sstream>


using namespace std;

void test_empty();
void test_size();
void test_height();
void test_copy();
void test_find();
void test_insert();
void test_min();
void test_max();
void test_check_sort();
void test_traverse();
void test_min_greater();

int main() {

	test_empty();
	test_size();
	test_height();
	test_copy();
	test_find();
	test_insert();
	test_min();
	test_max();
	test_check_sort();
	test_traverse();
	test_min_greater();

	return 0; 
}

void test_empty() {

	BinarySearchTree<int> tree;

	//testing empty tree
	assert(tree.empty());


	//testing one element
	tree.insert(2);
	assert(!tree.empty());

	//testing with 1+ elements
	tree.insert(3);
	tree.insert(4);
	tree.insert(5);
	assert(!tree.empty());

	cout << "test_empty pass" << endl;
	return;
}
void test_size() {
	BinarySearchTree<int> tree;

	//testing empty tree
	assert(tree.empty());
	assert(tree.size() == 0);


	//testing one element
	tree.insert(10);
	assert(!tree.empty());
	assert(tree.size() == 1);

	//testing with 1+ elements
	tree.insert(3);
	tree.insert(11);
	tree.insert(5);

	assert(!tree.empty());
	assert(tree.size() == 4);


	cout << "test_size pass" << endl;
	return;
}
void test_height() {

	BinarySearchTree<int> tree;

	//testing empty tree
	assert(tree.empty());
	assert(tree.size() == 0);
	assert(tree.height() == 0);


	//testing one element
	tree.insert(10);
	assert(!tree.empty());
	assert(tree.size() == 1);
	assert(tree.height() == 1);

	//testing with 1+ elements
	tree.insert(3);
	tree.insert(11);
	tree.insert(5);
	tree.insert(7);
	assert(tree.height() == 4);

	assert(!tree.empty());
	assert(tree.size() == 5);

	cout << "test_size pass" << endl;
	return;

}
void test_copy() {
	BinarySearchTree<int> tree;
	BinarySearchTree<int> tree_ctor(tree);
	BinarySearchTree<int> tree_assign = tree;

	//test copying an empty tree

	assert(tree_ctor.empty() && tree_assign.empty());
	assert(tree_ctor.size() == 0 && tree_assign.size() == 0);
	
	//test copying a tree with 1 element
	tree.insert(10);
	BinarySearchTree<int> tree_ctor2(tree);
	tree_assign = tree;

	assert(!(tree_ctor.empty() && tree_assign.empty()));
	assert(tree_ctor2.size() == 1 && tree_assign.size() == 1);

	//test copying a tree with 1+ elements
	tree.insert(3);
	tree.insert(11);
	tree.insert(5);
	tree.insert(7);

	tree_assign = tree;
	BinarySearchTree<int> tree_ctor3(tree);

	assert(!(tree_ctor.empty() && tree_assign.empty()));
	assert(tree_ctor3.size() == 5 && tree_assign.size() == 5);
	assert(tree_ctor3.height() == 4 && tree_assign.height() == 4);
	assert(tree.size() == 5);

	cout << "test_copy pass" << endl;
	return;
}
void test_find() {
	BinarySearchTree<int> tree;

	//testing empty tree
	assert(tree.empty());
	assert(tree.size() == 0);
	assert(tree.height() == 0);
	assert(tree.find(45) == tree.end());

	//testing one element
	tree.insert(10);
	assert(!tree.empty());
	assert(tree.size() == 1);
	assert(*tree.find(10) == 10);

	//testing with 1+ elements

	//for find_impl(), should we account for user error?
	tree.insert(3);
	tree.insert(11);
	*tree.find(11) = 5;
	assert(!tree.check_sorting_invariant());

	cout << "test_find pass" << endl;
	return;
}
void test_insert() {

	BinarySearchTree<int> tree;

	//testing empty tree
	assert(tree.empty());
	assert(tree.size() == 0);
	assert(tree.height() == 0);

	//testing one element
	tree.insert(10);
	assert(!tree.empty());
	assert(tree.size() == 1);

	//testing with 1+ elements
	tree.insert(3);
	tree.insert(11);
	tree.insert(5);
	tree.insert(7);
	assert(tree.height() == 4);
	assert(!tree.empty());
	assert(tree.size() == 5);

	assert(tree.check_sorting_invariant());

	cout << "test_insert pass" << endl;
	return;
}
void test_min() {
	BinarySearchTree<int> tree;

	//testing empty tree
	assert(tree.empty());
	assert(tree.size() == 0);
	assert(tree.height() == 0);

	//testing one element
	tree.insert(10);
	assert(!tree.empty());
	assert(tree.size() == 1);
	assert(*tree.min_element() == 10);


	//testing with 1+ elements
	tree.insert(3);

	tree.insert(11);
	tree.insert(5);
	tree.insert(7);
	assert(tree.height() == 4);
	assert(*tree.min_element() == 3);
	assert(!tree.empty());
	assert(tree.size() == 5);


	cout << "test_min pass" << endl;
	return;

}
void test_max() {
	BinarySearchTree<int> tree;

	//testing empty tree
	assert(tree.empty());
	assert(tree.size() == 0);
	assert(tree.height() == 0);

	//testing one element
	tree.insert(10);
	assert(!tree.empty());
	assert(tree.size() == 1);
	assert(*tree.max_element() == 10);


	//testing with 1+ elements
	tree.insert(3);

	tree.insert(11);
	tree.insert(5);
	tree.insert(7);
	assert(tree.height() == 4);
	assert(*tree.max_element() == 11);
	assert(!tree.empty());
	assert(tree.size() == 5);

	cout << "test_max pass" << endl;
	return; 
}
void test_check_sort() {
	BinarySearchTree<int> tree;

	//testing empty tree
	assert(tree.empty());
	assert(tree.check_sorting_invariant());

	//testing one element
	tree.insert(10);
	assert(tree.check_sorting_invariant());


	//testing with 1+ elements
	tree.insert(3);
	tree.insert(11);
	tree.insert(5);
	tree.insert(7);
	tree.insert(12);
	assert(tree.check_sorting_invariant());

	*tree.find(12) = 0;

	assert(!tree.check_sorting_invariant());

	cout << "test_check_sort pass" << endl;
	return;


}
void test_traverse() {
	BinarySearchTree<int> tree;

	//testing empty tree
	assert(tree.empty());
	assert(tree.check_sorting_invariant());

	//testing one element
	tree.insert(10);
	assert(tree.check_sorting_invariant());


	//testing with 1+ elements
	tree.insert(3);
	tree.insert(11);
	tree.insert(5);
	tree.insert(7);
	tree.insert(12);

	ostringstream oss_preorder;
	tree.traverse_preorder(oss_preorder);
	assert(oss_preorder.str() == "10 3 5 7 11 12 ");

	ostringstream oss_inorder;
	tree.traverse_inorder(oss_inorder);
	assert(oss_inorder.str() == "3 5 7 10 11 12 ");

	cout << "test_traverse pass" << endl;
	return;
}
void test_min_greater() {
	BinarySearchTree<int> tree;

	//testing empty tree
	assert(tree.empty());
	assert(tree.check_sorting_invariant());

	//testing one element
	tree.insert(10);
	assert(tree.check_sorting_invariant());


	//testing with 1+ elements
	tree.insert(3);
	tree.insert(11);
	tree.insert(5);
	tree.insert(7);
	tree.insert(12);

	assert(*tree.min_greater_than(2) == 3);
	assert(*tree.min_greater_than(3) == 5);
	assert(*tree.min_greater_than(7) == 10);
	assert(*tree.min_greater_than(10) == 11);
	assert(*tree.min_greater_than(11) == 12);

	cout << "test_min_greater() pass" << endl;
	return;
}