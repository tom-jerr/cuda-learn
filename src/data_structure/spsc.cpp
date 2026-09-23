#include <atomic>
#include <cstddef>
/**
 * @brief 这个实现完全无锁，甚至无等待（wait-free）：
          生产者和消费者各写各的指针，互不干扰。
          不需要 CAS，只需要 load/store + 内存序。
          这是音视频、网络、日志等领域的标准做法
 * @tparam T
 * @tparam N
 */
template <typename T, size_t N> class SPSCQueue {
  T buffer[N];
  std::atomic<size_t> head_{0};
  std::atomic<size_t> tail_{0};

public:
  bool push(const T &v) {
    size_t tail = tail_.load(std::memory_order_relaxed);
    size_t next = (tail + 1) % N;
    if (next == head_.load(std::memory_order_acquire))
      return false;
    buffer[tail] = v;
    tail_.store(next, std::memory_order_release);
  }
  bool pop(T &v) {
    size_t head = head_.load(std::memory_order_relaxed);
    size_t tail = tail_.load(std::memory_order_acquire);
    if (head == tail)
      return false; // 空
    v = buffer[head];
    head_.store((head + 1) % N, std::memory_order_release);
    return true;
  }
};