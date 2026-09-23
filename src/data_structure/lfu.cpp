#include <unordered_map>

using namespace std;
struct Node {
  int key, val, freq;
  Node *prev;
  Node *next;

  Node(int k, int v, int f)
      : key(k), val(v), freq(f), prev(nullptr), next(nullptr) {}
};
class FreqList {
public:
  Node *dummy;
  int size;

  FreqList() {
    dummy = new Node(0, 0, 0);
    dummy->next = dummy;
    dummy->prev = dummy;
    size = 0;
  }

  // 将节点添加到链表头部（表示最近使用）
  void addToHead(Node *node) {
    node->prev = dummy;
    node->next = dummy->next;
    dummy->next->prev = node;
    dummy->next = node;
    size++;
  }

  // 从链表中移除指定节点
  void removeNode(Node *node) {
    node->prev->next = node->next;
    node->next->prev = node->prev;
    size--;
  }

  // 判断链表是否为空
  bool isEmpty() { return size == 0; }

  // 获取链表尾部节点（最久未使用）
  Node *getTail() { return dummy->prev; }
};

class LFU {
  int capacity;
  int minFreq{0};
  unordered_map<int, Node *> map;
  unordered_map<int, FreqList *> lists;

  void updateLists(Node *node) {
    int oldfreq = node->freq;
    FreqList *olist = lists[oldfreq];

    olist->removeNode(node);
    if (olist->isEmpty()) {
      lists.erase(oldfreq);
      delete olist;
      if (minFreq == oldfreq)
        minFreq++;
    }

    node->freq++;
    int newFreq = node->freq;
    if (lists.find(newFreq) == lists.end())
      lists[newFreq] = new FreqList();
    lists[newFreq]->addToHead(node);
  }

public:
  LFU(int cap) : capacity(cap), minFreq(0) {}

  void insert(int key, int val) {
    auto it = map.find(key);
    if (it != map.end()) {
      Node *node = it->second;
      node->val = val;
      updateLists(node);
      return;
    }
    if (map.size() == capacity) {
      FreqList *removelist = lists[minFreq];
      Node *evictNode = removelist->getTail();
      removelist->removeNode(evictNode);
      map.erase(evictNode->key);
      delete evictNode;

      if (removelist->isEmpty()) {
        lists.erase(minFreq);
        delete removelist;
      }
    }

    Node *newNode = new Node(key, val, 1); // 初始频率是 1
    if (lists.find(1) == lists.end())
      lists[1] = new FreqList();
    lists[1]->addToHead(newNode);
    map[key] = newNode;
    minFreq = 1;
  }

  int get(int key) {
    auto it = map.find(key);
    if (it == map.end())
      return -1;
    updateLists(it->second);
    return it->second->val;
  }
};
