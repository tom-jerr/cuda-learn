#include <atomic>
#include <memory>
template <typename T> class MPMCQueue {
  struct Node {
    std::atomic<Node *> next{nullptr};
    T value;
    Node() : next(nullptr) {}
    explicit Node(T v) : next(nullptr), value(std::move(v)) {}
  };
  std::atomic<Node *> head_;
  std::atomic<Node *> tail_;

public:
  MPMCQueue() {
    Node *dummy = new Node();
    head_.store(dummy, std::memory_order_relaxed);
    tail_.store(dummy, std::memory_order_relaxed);
  }

  void enqueue(T value) {
    Node *node = new Node(std::move(value));
    Node *tail = nullptr;
    Node *next = nullptr;

    while (true) {
      tail = tail_.load(std::memory_order_acquire);
      next = tail->next.load(std::memory_order_acquire);

      // tail 不是真正的尾节点，说明别的线程已经插入了新节点但没更新 tail_
      if (next != nullptr) {
        // 帮助推进 tail_，这是一种“协作式”机制
        tail_.compare_exchange_weak(tail, next, std::memory_order_release,
                                    std::memory_order_relaxed);
        continue;
      }

      // 尝试把新节点挂到 tail->next 上
      if (tail->next.compare_exchange_weak(next, node,
                                           std::memory_order_release,
                                           std::memory_order_relaxed)) {
        break; // 挂链成功
      }
      // 失败说明别的线程抢先挂上了，重试
    }

    // 尝试把 tail_ 推进到新节点（失败也没关系，别的线程会帮忙推进）
    tail_.compare_exchange_weak(tail, node, std::memory_order_release,
                                std::memory_order_relaxed);
  }

  bool dequeue(T &out) {
    Node *head = nullptr;
    Node *tail = nullptr;
    Node *next = nullptr;

    while (true) {
      head = head_.load(std::memory_order_acquire);
      tail = tail_.load(std::memory_order_acquire);
      next = head->next.load(std::memory_order_acquire);

      // 队列空
      if (next == nullptr) {
        // 如果 tail_ 落后了，帮忙推进一下
        if (head == tail)
          return false;
        // 否则 tail_ 已经过期，重试
        continue;
      }

      // 如果 head 和 tail 不相等但 next 非空，说明 head 该被推进了
      // 先尝试把 head_ 从 head 推进到 next
      if (head == tail) {
        // 帮助推进 tail_，防止其他线程看到落后的 tail_
        tail_.compare_exchange_weak(tail, next, std::memory_order_release,
                                    std::memory_order_relaxed);
        // 注意：这里不返回，继续循环，因为还可能没有数据
        continue;
      }

      // 到这里 head != tail，next 一定非空，可以安全抢 head
      if (head_.compare_exchange_weak(head, next, std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
        // 抢到了，读取数据并删除旧 dummy 节点
        out = std::move(next->value);
        delete head; // head 是被淘汰的 dummy 节点
        return true;
      }
      // 失败重试
    }
  }
};