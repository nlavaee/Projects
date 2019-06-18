#include "BinarySearchTree.h"
#include <iostream>
#include <sstream>

using namespace std;

void test_empty_tree(); 
void test_one_elm_tree_right();
void test_one_elm_tree_left();
void test_two_elm_right_left();

int main() {

	test_empty_tree();
	test_one_elm_tree_right();
	test_one_elm_tree_left();
	test_two_elm_right_left();

}

//testing empty tree
void test_empty_tree() {
	BinarySearchTree<int> tree;

	//making sure tree is empty
	assert(tree.empty());
	assert(tree.size() == 0);
	assert(tree.height() == 0);
	assert(tree.min_element() == tree.end());
	assert(tree.max_element() == tree.end());
	assert(tree.check_sorting_invariant());
	assert(tree.min_greater_than(0) == tree.end());
	assert(tree.find(5) == tree.end());

	//printing an empty tree - preorder
	ostringstream oss_preorder;
	tree.traverse_preorder(oss_preorder);
	assert(oss_preorder.str() == "");

	//printing an empty tree - inorder
	ostringstream oss_inorder;
	tree.traverse_inorder(oss_inorder);
	assert(oss_inorder.str() == "");

	cout << "test_empty_tree pass" << endl;

	return;
}

//testing various trees with different types that have only one element
void test_one_elm_tree_right() {
	//----------------------------------------
	// TESTING WITH INTS
	//----------------------------------------
	BinarySearchTree<int> tree_int;
	tree_int.insert(1);

	//making sure tree has one value
	assert(!tree_int.empty());
	assert(tree_int.size() == 1);
	assert(tree_int.height() == 1);
	assert(*tree_int.min_element() == 1);
	assert(*tree_int.max_element() == 1);
	assert(tree_int.check_sorting_invariant());
	assert(tree_int.min_greater_than(5) == tree_int.end());
	assert(tree_int.find(5) == tree_int.end());
	assert(*tree_int.find(1) == 1);

	//printing an empty tree - preorder
	ostringstream oss_preorder_int;
	tree_int.traverse_preorder(oss_preorder_int);
	assert(oss_preorder_int.str() == "1 ");

	//printing an empty tree - inorder
	ostringstream oss_inorder_int;
	tree_int.traverse_inorder(oss_inorder_int);
	assert(oss_inorder_int.str() == "1 ");

	//------------------------------------------
	// TESTING WITH STRINGS
	//------------------------------------------

	BinarySearchTree<string> tree_string;
	tree_string.insert("klaus");

	//making sure tree has one value
	assert(!tree_string.empty());
	assert(tree_string.size() == 1);
	assert(tree_string.height() == 1);
	assert(*tree_string.min_element() == "klaus");
	assert(*tree_string.max_element() == "klaus");
	assert(tree_string.check_sorting_invariant());
	assert(tree_string.min_greater_than("zebra") == tree_string.end());
	assert(tree_string.find("eggs") == tree_string.end());
	assert(*tree_string.find("klaus") =="klaus");

	//printing an empty tree - preorder
	ostringstream oss_preorder_string;
	tree_string.traverse_preorder(oss_preorder_string);

	assert(oss_preorder_string.str() == "klaus ");

	//printing an empty tree - inorder
	ostringstream oss_inorder_string;
	tree_string.traverse_inorder(oss_inorder_string);

	assert(oss_inorder_string.str() == "klaus ");

	//------------------------------------------
	// TESTING WITH CHARS
	//------------------------------------------

	BinarySearchTree<char> tree_char;
	tree_char.insert('a');

	//making sure tree has one value
	assert(!tree_char.empty());
	assert(tree_char.size() == 1);
	assert(tree_char.height() == 1);
	assert(*tree_char.min_element() == 'a');
	assert(*tree_char.max_element() == 'a');
	assert(tree_char.check_sorting_invariant());
	assert(tree_char.min_greater_than('b') == tree_char.end());

	//printing an empty tree - preorder
	ostringstream oss_preorder_char;
	tree_char.traverse_preorder(oss_preorder_char);
	assert(oss_preorder_char.str() == "a ");

	//printing an empty tree - inorder
	ostringstream oss_inorder_char;
	tree_char.traverse_inorder(oss_inorder_char);
	assert(oss_inorder_char.str() == "a ");

	cout << "test_one_elm_tree_right pass" << endl;
	return;
}

//testing one element in left branch + root
void 	test_one_elm_tree_left() {
	//----------------------------------------
	// TESTING WITH INTS
	//----------------------------------------
	BinarySearchTree<int> tree_int;
	tree_int.insert(3);

	//making sure tree has one value
	assert(!tree_int.empty());
	assert(tree_int.size() == 1);
	assert(tree_int.height() == 1);
	assert(*tree_int.min_element() == 3);
	assert(*tree_int.max_element() == 3);
	assert(tree_int.check_sorting_invariant());
	assert(*tree_int.min_greater_than(1) == 3);

	//printing an empty tree - preorder
	ostringstream oss_preorder_int;
	tree_int.traverse_preorder(oss_preorder_int);
	assert(oss_preorder_int.str() == "3 ");

	//printing an empty tree - inorder
	ostringstream oss_inorder_int;
	tree_int.traverse_inorder(oss_inorder_int);
	assert(oss_inorder_int.str() == "3 ");

	//------------------------------------------
	// TESTING WITH STRINGS
	//------------------------------------------

	BinarySearchTree<string> tree_string;
	tree_string.insert("klaus");

	//making sure tree has one value
	assert(!tree_string.empty());
	assert(tree_string.size() == 1);
	assert(tree_string.height() == 1);
	assert(*tree_string.min_element() == "klaus");
	assert(*tree_string.max_element() == "klaus");
	assert(tree_string.check_sorting_invariant());
	assert(*tree_string.min_greater_than("elijah") == "klaus");

	//printing an empty tree - preorder
	ostringstream oss_preorder_string;
	tree_string.traverse_preorder(oss_preorder_string);
	assert(oss_preorder_string.str() == "klaus ");

	//printing an empty tree - inorder
	ostringstream oss_inorder_string;
	tree_string.traverse_inorder(oss_inorder_string);
	assert(oss_inorder_string.str() == "klaus ");

	//------------------------------------------
	// TESTING WITH CHARS
	//------------------------------------------

	BinarySearchTree<char> tree_char;
	tree_char.insert('a');

	//making sure tree has one value
	assert(!tree_char.empty());
	assert(tree_char.size() == 1);
	assert(tree_char.height() == 1);
	assert(*tree_char.min_element() == 'a');
	assert(*tree_char.max_element() == 'a');
	assert(tree_char.check_sorting_invariant());
	assert(*tree_char.min_greater_than('A') == 'a');

	//printing an empty tree - preorder
	ostringstream oss_preorder_char;
	tree_char.traverse_preorder(oss_preorder_char);
	assert(oss_preorder_char.str() == "a ");

	//printing an empty tree - inorder
	ostringstream oss_inorder_char;
	tree_char.traverse_inorder(oss_inorder_char);
	assert(oss_inorder_char.str() == "a ");

	cout << "test_one_elm_tree_left pass" << endl;
	return;
}

//testing one element in right/left branch + root
void test_two_elm_right_left() {
	BinarySearchTree<int> tree;
	tree.insert(6); //node
	tree.insert(4); //node->left
	tree.insert(8); //node->right

	//making sure tree is empty
	assert(!tree.empty());
	assert(tree.size() == 3);
	assert(tree.height() == 2);
	assert(*tree.min_element() == 4);
	assert(*tree.max_element() == 8);
	assert(tree.check_sorting_invariant());
	assert(*tree.min_greater_than(0) == 4);
	assert(*tree.min_greater_than(4) == 6);
	assert(*tree.find(4) == 4);

	//printing an empty tree - preorder
	ostringstream oss_preorder;
	tree.traverse_preorder(oss_preorder);
	assert(oss_preorder.str() == "6 4 8 ");

	//printing an empty tree - inorder
	ostringstream oss_inorder;
	tree.traverse_inorder(oss_inorder);
	assert(oss_inorder.str() == "4 6 8 ");

	cout << "test_two_elm_right_left pass" << endl;
	return;
}