#include <algorithm>
#include <cassert>
#include <optional>
#include <unordered_map>

struct Node {
  int key;
  int val;
  // -1 表示永不过期，否则表示逻辑计数器到达该值时过期。
  long long expired;
  Node *prev;
  Node *next;

  Node(int k, int v, long long e = -1)
      : key(k), val(v), expired(e), prev(nullptr), next(nullptr) {}
};

class TTLLRU {
private:
  Node *dummy;
  std::unordered_map<int, Node *> map;
  int cap;
  int cnt;
  long long counter;

  void removeNode(Node *node) {
    node->prev->next = node->next;
    node->next->prev = node->prev;
  }

  void addToHead(Node *node) {
    node->next = dummy->next;
    node->prev = dummy;
    dummy->next->prev = node;
    dummy->next = node;
  }

  void moveToHead(Node *node) {
    removeNode(node);
    addToHead(node);
  }

  void eraseNode(Node *node) {
    removeNode(node);
    map.erase(node->key);
    delete node;
    --cnt;
  }

  void purgeExpired() {
    Node *node = dummy->next;
    while (node != dummy) {
      Node *next = node->next;
      if (node->expired != -1 && node->expired <= counter) {
        eraseNode(node);
      }
      node = next;
    }
  }

  void nextOperation() {
    ++counter;
    purgeExpired();
  }

public:
  explicit TTLLRU(int capacity)
      : dummy(new Node(-1, -1)), cap(std::max(0, capacity)), cnt(0),
        counter(0) {
    dummy->next = dummy;
    dummy->prev = dummy;
  }

  ~TTLLRU() {
    Node *node = dummy->next;
    while (node != dummy) {
      Node *next = node->next;
      delete node;
      node = next;
    }
    delete dummy;
  }

  TTLLRU(const TTLLRU &) = delete;
  TTLLRU &operator=(const TTLLRU &) = delete;

  // 每次 insert/get 都让 counter 加一。
  // ttl < 0 表示永不过期，ttl == 0 表示立即过期。
  void insert(int key, int val, long long ttl = -1) {
    nextOperation();

    auto it = map.find(key);
    if (ttl == 0) {
      if (it != map.end()) {
        eraseNode(it->second);
      }
      return;
    }

    const long long expired = ttl < 0 ? -1 : counter + ttl;
    if (it != map.end()) {
      it->second->val = val;
      it->second->expired = expired;
      moveToHead(it->second);
      return;
    }

    if (cap == 0) {
      return;
    }
    if (cnt == cap) {
      eraseNode(dummy->prev);
    }

    Node *node = new Node(key, val, expired);
    addToHead(node);
    map[key] = node;
    ++cnt;
  }

  std::optional<int> get(int key) {
    nextOperation();

    auto it = map.find(key);
    if (it == map.end()) {
      return std::nullopt;
    }
    moveToHead(it->second);
    return it->second->val;
  }

  // 手动推进逻辑时间，不进行缓存读写。
  void advance(long long steps = 1) {
    if (steps <= 0) {
      return;
    }
    counter += steps;
    purgeExpired();
  }

  int size() {
    purgeExpired();
    return cnt;
  }
};

#ifdef TTL_LRU_SELF_TEST
int main() {
  TTLLRU lru(2);
  lru.insert(1, 10);
  lru.insert(2, 20);
  assert(lru.get(1) == 10);
  lru.insert(3, 30);
  assert(!lru.get(2));
  assert(lru.get(1) == 10);
  assert(lru.get(3) == 30);

  TTLLRU ttl(2);
  ttl.insert(1, 10, 2);
  assert(ttl.get(1) == 10);
  assert(!ttl.get(1));

  TTLLRU expiration_first(2);
  expiration_first.insert(1, 10, 2);
  expiration_first.insert(2, 20);
  expiration_first.insert(3, 30);
  assert(expiration_first.get(2) == 20);
  assert(expiration_first.get(3) == 30);

  TTLLRU disabled(0);
  disabled.insert(1, 1);
  assert(disabled.size() == 0);
}
#endif
