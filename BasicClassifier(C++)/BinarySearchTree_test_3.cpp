#include "BinarySearchTree.h"
#include <iostream>
#include <sstream>

using namespace std;

void test_check_sort_left();
void test_check_sort_right();
void test_check_sort_both();
void test_check_sort_both_breaking();
void test_traverse();
void test_size();
void test_height();

int main() {

	test_check_sort_left();
	test_check_sort_right();
	test_check_sort_both();
	test_check_sort_both_breaking();
	cout << "check_sort_tests pass " << endl;

	test_traverse();
	cout << "traverse_tests pass " << endl;

	test_size();
	cout << "size_test pass" << endl;

	test_height();
	cout << "height_test pass" << endl;
	return 0;
}

//Testing Checking Sort Invariant
//	- when it's empty
//  - when there's only one node
//  - when there is only one node to the left / right
//  - when there are only left nodes
//  - when there are only right nodes
//  - when it branches only on the left / right
//  - many elements
//   - breaking the variant / when tree is copied
void test_check_sort_left() {

	BinarySearchTree<int> tree;
	
	

	tree.insert(10);
	

	tree.insert(7);
	
	 
	tree.insert(5);
	tree.insert(3);
	tree.insert(1);
	

	tree.insert(8);
	tree.insert(6);
	tree.insert(4);
	tree.insert(2);
	tree.insert(0);

	

	

	return;
}
void test_check_sort_right() {

	BinarySearchTree<int> tree;

	

	tree.insert(10);
	

	tree.insert(12);
	

	tree.insert(16);
	tree.insert(18);
	tree.insert(20);
	

	tree.insert(11);
	tree.insert(13);
	tree.insert(14);
	tree.insert(15);
	tree.insert(19);

	

	return;
}
void test_check_sort_both() {

	BinarySearchTree<int> tree;

	

	tree.insert(10);
	

	tree.insert(12);
	

	tree.insert(16);
	tree.insert(18);
	tree.insert(20);
	

	tree.insert(11);
	tree.insert(13);
	tree.insert(14);
	tree.insert(15);
	tree.insert(19);

	tree.insert(7);
	

	tree.insert(5);
	tree.insert(3);
	tree.insert(1);
	

	tree.insert(8);
	tree.insert(6);
	tree.insert(4);
	tree.insert(2);
	tree.insert(0);

	

	return;
}
void test_check_sort_both_breaking() {

	BinarySearchTree<int> tree;

	

	tree.insert(10);
	

	tree.insert(12);
	

	tree.insert(16);
	tree.insert(18);
	tree.insert(20);
	

	tree.insert(11);
	tree.insert(13);
	tree.insert(14);
	tree.insert(15);
	tree.insert(19);

	tree.insert(7);
	

	tree.insert(5);
	tree.insert(3);
	tree.insert(1);
	

	tree.insert(8);
	tree.insert(6);
	tree.insert(4);
	tree.insert(2);
	tree.insert(0);

	

	BinarySearchTree<int> tree2 = tree;
	BinarySearchTree<int> tree3(tree2);

	assert(tree2.check_sorting_invariant());
	assert(tree3.check_sorting_invariant());

	*tree.find(5) = 53;
	assert(!tree.check_sorting_invariant());

	*tree.find(20) = -5;
	assert(!tree.check_sorting_invariant());

	*tree2.find(5) = 53;
	assert(!tree2.check_sorting_invariant());

	*tree3.find(20) = -5;
	assert(!tree3.check_sorting_invariant());

	return;
}

void test_traverse() {

	ostringstream oss_inorder;
	ostringstream oss_preorder;

	BinarySearchTree<int> tree;

	tree.traverse_preorder(oss_preorder);
	assert(oss_preorder.str() == "");

	tree.traverse_inorder(oss_inorder);
	assert(oss_inorder.str() == "");

	tree.insert(10);
	
	oss_preorder.str("");
	oss_inorder.str("");
	tree.traverse_preorder(oss_preorder);
	assert(oss_preorder.str() == "10 ");

	tree.traverse_inorder(oss_inorder);
	assert(oss_inorder.str() == "10 ");

	tree.insert(7);

	oss_preorder.str("");
	oss_inorder.str("");
	tree.traverse_preorder(oss_preorder);
	assert(oss_preorder.str() == "10 7 ");

	tree.traverse_inorder(oss_inorder);
	assert(oss_inorder.str() == "7 10 ");

	tree.insert(5);
	tree.insert(3);
	tree.insert(1);

	oss_preorder.str("");
	oss_inorder.str("");
	tree.traverse_preorder(oss_preorder);
	assert(oss_preorder.str() == "10 7 5 3 1 ");

	tree.traverse_inorder(oss_inorder);
	assert(oss_inorder.str() == "1 3 5 7 10 ");

	tree.insert(8);
	tree.insert(6);
	tree.insert(4);
	tree.insert(2);
	tree.insert(0);

	oss_preorder.str("");
	oss_inorder.str("");
	tree.traverse_preorder(oss_preorder);
	assert(oss_preorder.str() == "10 7 5 3 1 0 2 4 6 8 ");

	tree.traverse_inorder(oss_inorder);
	assert(oss_inorder.str() == "0 1 2 3 4 5 6 7 8 10 ");

	oss_preorder.str("");
	oss_inorder.str("");

	return;
}



//Testing Checking Sort Invariant
//	- when it's empty
//  - when there's only one node
//  - when there is only one node to the left / right
//  - when there are only left nodes
//  - when there are only right nodes
//  - when it branches only on the left / right
//  - when tree is copied
void test_size() {

	BinarySearchTree<int> tree_left;
	assert(tree_left.size() == 0);

	tree_left.insert(10);
	assert(tree_left.size() == 1);

	tree_left.insert(9);
	tree_left.insert(7);
	tree_left.insert(5);
	tree_left.insert(2);
	assert(tree_left.size() == 5);

	tree_left.insert(8);
	tree_left.insert(6);
	tree_left.insert(3);
	assert(tree_left.size() == 8);

	BinarySearchTree<int> tree_right;
	assert(tree_right.size() == 0);

	tree_right.insert(11);
	assert(tree_right.size() == 1);

	tree_right.insert(12);
	tree_right.insert(17);
	tree_right.insert(55);
	tree_right.insert(72);
	assert(tree_right.size() == 5);

	tree_right.insert(18);
	tree_right.insert(16);
	tree_right.insert(13);
	assert(tree_right.size() == 8);

	BinarySearchTree<int> tree_both(tree_left);
	
	for (auto i : tree_right) {
		tree_both.insert(i);
	}

	assert(tree_both.size() == 16);

	return;
}

void test_height(){
	BinarySearchTree<int> tree_left;
	assert(tree_left.height() == 0);

	tree_left.insert(10);
	assert(tree_left.height() == 1);

	tree_left.insert(9);
	tree_left.insert(7);
	tree_left.insert(5);
	tree_left.insert(2);
	assert(tree_left.height() == 5);

	tree_left.insert(8);
	tree_left.insert(6);
	tree_left.insert(3);
	assert(tree_left.height() == 6);

	BinarySearchTree<int> tree_right;
	assert(tree_right.height() == 0);

	tree_right.insert(11);
	assert(tree_right.height() == 1);

	tree_right.insert(12);
	tree_right.insert(17);
	tree_right.insert(55);
	tree_right.insert(72);
	assert(tree_right.height() == 5);

	tree_right.insert(18);
	tree_right.insert(16);
	tree_right.insert(13);
	assert(tree_right.height() == 5);

	ostringstream oss_inorder;
	BinarySearchTree<int> tree_both(tree_left);

	for (auto i : tree_right) {
		tree_both.insert(i);
	}
	assert(tree_both.height() == 9);

	return;
}

