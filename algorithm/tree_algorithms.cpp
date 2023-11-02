#include <iostream>
#include <vector>
#include <stack>

template<class T>
struct node {
public:
  node():left(nullptr), right(nullptr) {}
  T value;
  node * left;
  node * right;
};

template<class T>
void DLR(node<T> * root) {
  std::stack<node<T> *> stack;
  stack.push(root);

  node<T> * top;
  while(!stack.empty()) {
    top = stack.top();
    std::cout << top->value << std::endl;
    stack.pop();
    if (top->right != nullptr) stack.push(top->right);
    if (top->left != nullptr) stack.push(top->left);
  }
  return;
};

template<class T>
void LDR(node<T> * root) {
  std::stack<node<T> *> stack;
  while(root != nullptr || !stack.empty()) {
    while(root != nullptr) {
      stack.push(root);
      root = root->left;
    }
    if(!stack.empty()){
      root = stack.top();
      std::cout << root->value << std::endl;
      stack.pop();
      root = root->right;
    }
  }
  return;
};

template<class T>
void LRD(node<T> * root) {
  if (root == nullptr) return;
  std::stack<node<T> *> stack;
  stack.push(root);

  node<T> * last_pop = nullptr;
  while(!stack.empty()) {
    while(root->left != nullptr) {
      root = root->left;
      stack.push(root);
    }
    while(!stack.empty()) {
      if (root->right == nullptr || root->right == last_pop) {
        std::cout << root->value << std::endl;
        stack.pop();
        last_pop = root;
        if (!stack.empty()){
          root = stack.top();
        }
      } else {
        root = root->right;
        stack.push(root);
        break;
      }
    }
  }
  return;
};

/*
        1
       / \
      2   3
       \
        4
       /
      5
*/

int main() {
  node<int> * root = new node<int>();
  root->value = 1;
  root->left = new node<int>();
  root->left->value = 2;
  root->right = new node<int>();
  root->right->value = 3;
  root->left->right = new node<int>();
  root->left->right->value = 4;
  root->left->right->left = new node<int>();
  root->left->right->left->value = 5;

  DLR(root);
  LDR(root);
  LRD(root);

  return 0;
}
