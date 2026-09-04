#include <string>
#include <unordered_map>
#include <vector>
using namespace std;
struct Node {
  bool end = false;
  unordered_map<char, pair<string, Node *>> map;
};

class RadixTree {
  Node *root;
  static size_t commonPrefixLength(const string &edge, const string &tokens,
                                   size_t pos) {
    size_t len = 0;

    while (len < edge.size() && pos + len < tokens.size() &&
           edge[len] == tokens[pos + len]) {
      ++len;
    }

    return len;
  }

  RadixTree() : root(new Node()) {}
  void insert(string tokens) {
    char ch = tokens[0];

    int pos = 0;
    int n = tokens.size();
    Node *cur = root;
    while (pos < n) {
      auto it = root->map.find(ch);
      if (it == root->map.end()) {
        Node *newNode = new Node();
        root->map[0] = {tokens, newNode};
        return;
      }
      string edge = it->second.first;
      Node *child = it->second.second;
      int common = commonPrefixLength(edge, tokens, pos);
      if (common == edge.size()) {
        pos += common;
        cur = child;
        continue;
      }
      // match some
      string commonPrefix = edge.substr(0, common);
      string oldSuffix = edge.substr(common);
      Node *middle = new Node();
      it->second = {commonPrefix, middle};
      middle->map[oldSuffix[0]] = {oldSuffix, child};
      pos += common;

      if (pos == n) {
        middle->end = true;
        return;
      }

      string newSuffix = tokens.substr(pos);
      Node *newNode = new Node();
      newNode->end = true;
      middle->map[newSuffix[0]] = {newSuffix, newNode};
      return;
    }
    cur->end = true;
  }

  pair<string, Node *> match_prefix(string tokens) {
    Node *cur = root;
    Node *res = nullptr;
    int pos = 0;
    int n = tokens.size();

    while (pos < tokens.size()) {
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