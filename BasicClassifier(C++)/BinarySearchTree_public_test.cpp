#include "BinarySearchTree.h"
#include <sstream>

using namespace std;

int main() {
  cout << "starting test" << endl;
  BinarySearchTree<int> tree;

  tree.insert(5);

  assert(tree.size() == 1);
  assert(tree.height() == 1);

  assert(tree.find(5) != tree.end());

  tree.insert(7);
  tree.insert(3);

  assert(tree.check_sorting_invariant());
  assert(*tree.max_element() == 7);
  assert(*tree.min_element() == 3);
  assert(*tree.min_greater_than(5) == 7);

  cout << "cout << tree.to_string()" << endl;
  cout << tree.to_string() << endl << endl;

  cout << "cout << tree" << endl << "(uses iterators)" << endl;
  cout << tree << endl << endl;

  ostringstream oss_preorder;
  tree.traverse_preorder(oss_preorder);
  cout << "preorder" << endl;
  cout << oss_preorder.str() << endl << endl;
  assert(oss_preorder.str() == "5 3 7 ");

  ostringstream oss_inorder;
  tree.traverse_inorder(oss_inorder);
  cout << "inorder" << endl;
  cout << oss_inorder.str() << endl << endl;
  assert(oss_inorder.str() == "3 5 7 ");

  cout << "BinarySearchTree_public_test PASS" << endl;
  cout << "Now it's your turn. Write more tests of your own!" << endl;
}
