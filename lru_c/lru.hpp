#ifndef _LRU_H_
#define _LRU_H_
#include <unordered_map>
#include <stack>
#include <vector>
#include <pybind11/pybind11.h>
namespace py = pybind11;
struct LinkNode {
    int key;
    int value;
    LinkNode *prev;
    LinkNode *next;
    LinkNode(int k = -1, int v = -1) : key(k), value(v), prev(nullptr), next(nullptr) {}
};
enum class OPType {
    Get,
    Add,
    Overflow
};
struct Op {
    OPType op_type;
    LinkNode *prev_node;
    LinkNode *next_node;
    LinkNode *removed_node;
    int key;
    int value;
    Op(OPType op = OPType::Add, LinkNode *prev = nullptr, LinkNode *next = nullptr, LinkNode *cur = nullptr, int k = -1, int v = -1) : op_type(op), prev_node(prev), next_node(next), removed_node(cur), key(k), value(v) {}
};
class LRU {
public:
    LRU(int capacity);
    int TryGet(int key);
    int Get(int key);
    int View(int key);
    void Rollback(int steps);
    std::vector<int> Keys();
    std::vector<int> Values();
    bool Contains(int key);
    py::list state_dict();
    void restore(py::list kvs);
    ~LRU();
    void Show();
    
private:
    int max_size_;
    int cur_size_;
    std::unordered_map<int, LinkNode *> cache_;
    std::stack<Op> op_records_;
    LinkNode *head_;
    LinkNode *tail_;
    // ObjectPool<LinkNode> memory_manager_;
};
#endif
