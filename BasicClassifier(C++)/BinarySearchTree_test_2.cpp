#include "BinarySearchTree.h"
#include <iostream>
#include <sstream>

using namespace std;

void test_left_stick();
void test_right_stick();
void test_left_stick_with_right();
void test_right_stick_with_left();

int main() {

	test_left_stick();
	test_right_stick();
	test_left_stick_with_right();
	test_right_stick_with_left();
	

	return 0;
}

void test_left_stick(){
	BinarySearchTree<int> tree;

	tree.insert(5);
	tree.insert(4);
	tree.insert(3);
	tree.insert(2);
	tree.insert(1);
	tree.insert(0);
	tree.insert(-1);
	tree.insert(-2);

	assert(!tree.empty());
	assert(tree.size() == 8);
	assert(tree.height() == 8);
	assert(*tree.min_element() == -2);
	assert(*tree.max_element() == 5);
	assert(tree.check_sorting_invariant());
	assert(tree.min_greater_than(5) == tree.end());
	assert(*tree.min_greater_than(-2) == -1);
	assert(tree.find(6) == tree.end());
	*tree.find(-2) = -3;
	assert(tree.check_sorting_invariant());
	assert(*tree.find(-3) == -3);

	//printing an empty tree - preorder
	ostringstream oss_preorder;
	tree.traverse_preorder(oss_preorder);
	assert(oss_preorder.str() == "5 4 3 2 1 0 -1 -3 ");

	//printing an empty tree - inorder
	ostringstream oss_inorder;
	tree.traverse_inorder(oss_inorder);
	assert(oss_inorder.str() == "-3 -1 0 1 2 3 4 5 ");


	cout << "test_left_stick pass" << endl; 
	return;
}
void test_right_stick(){

	BinarySearchTree<int> tree;

	tree.insert(5);
	tree.insert(6);
	tree.insert(7);
	tree.insert(8);
	tree.insert(9);
	tree.insert(10);
	tree.insert(11);
	tree.insert(12);

	assert(!tree.empty());
	assert(tree.size() == 8);
	assert(tree.height() == 8);
	assert(*tree.min_element() == 5);
	assert(*tree.max_element() == 12);
	assert(tree.check_sorting_invariant());
	assert(*tree.min_greater_than(5) == 6);
	assert(tree.min_greater_than(12) == tree.end());
	assert(*tree.find(6) == 6);
	*tree.find(12) = 23;
	assert(tree.check_sorting_invariant());
	assert(*tree.find(23) == 23);


	//printing an empty tree - preorder
	ostringstream oss_preorder;
	tree.traverse_preorder(oss_preorder);
	assert(oss_preorder.str() == "5 6 7 8 9 10 11 23 ");

	//printing an empty tree - inorder
	ostringstream oss_inorder;
	tree.traverse_inorder(oss_inorder);
	assert(oss_inorder.str() == "5 6 7 8 9 10 11 23 ");

	cout << "test_right_stick pass"<< endl; 
	return;
}
void test_left_stick_with_right(){

	BinarySearchTree<int> tree;

	tree.insert(15);
	tree.insert(13);
	tree.insert(10);
	tree.insert(5);
	tree.insert(1);
	tree.insert(-1);
	tree.insert(0);
	tree.insert(-10);

	assert(!tree.empty());
	assert(tree.size() == 8);
	assert(tree.height() == 7);
	assert(*tree.min_element() == -10);
	assert(*tree.max_element() == 15);
	assert(tree.check_sorting_invariant());
	assert(tree.min_greater_than(15) == tree.end());
	assert(*tree.min_greater_than(-2) == -1);
	assert(tree.find(6) == tree.end());
	assert(tree.check_sorting_invariant());

	//printing an empty tree - preorder
	ostringstream oss_preorder;
	tree.traverse_preorder(oss_preorder);
	assert(oss_preorder.str() == "15 13 10 5 1 -1 -10 0 ");

	//printing an empty tree - inorder
	ostringstream oss_inorder;
	tree.traverse_inorder(oss_inorder);
	assert(oss_inorder.str() == "-10 -1 0 1 5 10 13 15 ");

	cout << "test_left_stick_with_right pass" << endl; 
	return;
}
void test_right_stick_with_left(){
	
	BinarySearchTree<int> tree;

	tree.insert(15);
	tree.insert(23);
	tree.insert(30);
	tree.insert(45);
	tree.insert(61);
	tree.insert(71);
	tree.insert(60);
	tree.insert(62);

	assert(!tree.empty());
	assert(tree.size() == 8);
	assert(tree.height() == 7);
	assert(*tree.min_element() == 15);
	assert(*tree.max_element() == 71);
	assert(tree.check_sorting_invariant());
	assert(*tree.min_greater_than(15) == 23);
	assert(tree.check_sorting_invariant());

	//printing an empty tree - preorder
	ostringstream oss_preorder;
	tree.traverse_preorder(oss_preorder);
	assert(oss_preorder.str() == "15 23 30 45 61 60 71 62 ");

	//printing an empty tree - inorder
	ostringstream oss_inorder;
	tree.traverse_inorder(oss_inorder);
	assert(oss_inorder.str() == "15 23 30 45 60 61 62 71 ");

	cout << "test_right_stick_with_left pass" << endl; 
	return;
}