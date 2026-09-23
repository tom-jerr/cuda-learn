#include <atomic>

class Stack {
  struct Node {
    int val;
    Node *next;
    Node(int val) : val(val), next(nullptr) {}
  };

  std::atomic<Node *> head_{nullptr};

public:
  Stack() {}

  void push(int val) {
    Node *head = head_.load(std::memory_order_relaxed);
    Node *node = new Node(val);

    do {
      node->next = head;
    } while (!head_.compare_exchange_weak(head, node, std::memory_order_release,
                                          std::memory_order_relaxed));
  }

  bool pop(int &val) {
    Node *old_head = head_.load(std::memory_order_acquire);
    while (true) {
      if (head_.compare_exchange_weak(old_head, old_head->next,
                                      std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
        val = old_head->val;
        delete old_head;
        return true;
      }
    }
    return false; // empty
  }
};