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
}

void DLR(node * root) {
  std::stack<node *> stack;
  stack.push(root);

  node * top;
  while(!stack.empty()) {
    top = stack.top();
    std::cout << top->value << std::endl;
    stack.pop()
    if (top->right != nullptr) stack.push(top->right);
    if (top->left != nullptr) stack.push(top->left);
  }
  return;
}

void LDR(node * root) {
  std::stack<node *> stack;
  while(root != nullptr || !stack.empty()) {
    while(root != nullptr) {
      stack.push(root);
      root = root->left;
    }
    if(!stack.empty()){
      root = stack.top();
      std::cout << root->val << std::endl;
      stack.pop();
      root = root->right;
    }
  }
  return;
}

void LRD(node * root) {
  if (root == nullptr) return;
  std::stack<node*> stack;
  stack.push(root);

  node * last_pop = nullptr;
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
        root = stack.top();
      } else {
        root = root->right;
        stack.push(right);
        break;
      }
    }
  }
  return;
}


void topological_sorting() {

}

int main() {
}
