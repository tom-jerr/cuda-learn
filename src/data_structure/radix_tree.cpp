#include <string>
#include <unordered_map>
#include <vector>
using namespace std;
struct Node {
  bool end = false;
  unordered_map<char, pair<string, Node *>> map;
};

int lcp(const string &edge, const string &tokens, int pos) {
  int len = 0;
  while (len < edge.size() && pos + len < tokens.size() &&
         edge[len] == tokens[pos + len])
    len++;
  return len;
}

class RadixTree {
  Node *root;
  RadixTree() : root(new Node()) {}

  void insert(string tokens) {
    Node *cur = root;
    int pos = 0;
    while (pos < tokens.size()) {
      auto it = cur->map.find(tokens[pos]);
      // no prefix
      if (it == cur->map.end()) {
        cur->map[tokens[pos]] = {tokens, new Node()};
        return;
      }

      string prefix = cur->map[tokens[0]].first;
      Node *child = cur->map[tokens[0]].second;
      int len = lcp(prefix, tokens, pos);
      // all prefix, next node
      if (len == prefix.size()) {
        pos += len;
        cur = child;
        continue;
      }
      // some prefix
      string common = prefix.substr(pos, len);
      string old = prefix.substr(len);
      Node *middle = new Node();
      cur->map[common[0]] = {common, middle};
      middle->map[old[0]] = {old, child};
      pos += len;
      if (pos == tokens.size()) {
        middle->end = true;
        return;
      } else {
        string newtoken = tokens.substr(pos);
        Node *newNode = new Node();
        newNode->end = true;
        middle->map[newtoken[0]] = {newtoken, newNode};
        return;
      }
    }
    cur->end = true;
  }

  pair<string, Node *> match_prefix(string tokens) {
    Node *cur = root;
    Node *res = nullptr;
    int pos = 0;
    int n = tokens.size();

    while (pos < n) {
      auto it = cur->map.find(tokens[pos]);

      // 没有对应首字符的边，停止
      if (it == cur->map.end()) {
        break;
      }

      const string &edge = it->second.first;
      Node *child = it->second.second;

      size_t common = 0;

      // 逐字符寻找公共前缀
      while (common < edge.size() && pos + common < tokens.size()) {
        if (edge[common] != tokens[pos + common]) {
          break;
        }

        ++common;
      }

      pos += common;

      // 这一条边没有完整匹配，不能进入 child
      if (common != edge.size()) {
        break;
      }

      // 整条边匹配完成，才进入下一节点
      cur = child;
    }

    return {tokens.substr(0, pos), cur};
  }
};