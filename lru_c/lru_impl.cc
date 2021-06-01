#include "lru.hpp"
#include <iostream>
LRU::LRU(int capcity) {
    max_size_ = capcity;
    cur_size_ = 0;
    // head_ = memory_manager_.allocate();
    // tail_ = memory_manager_.allocate();
    head_ = new LinkNode();
    tail_ = new LinkNode();
    head_->next = tail_;
    tail_->prev = head_;
}
LRU::~LRU() {
    LinkNode* cur = head_->next;
    while(cur != tail_) {
        auto next = cur->next;
        delete cur;
        cur = next;
    }
    delete head_;
    delete tail_;
}
int LRU::Get(int key) {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
        LinkNode *new_node = new LinkNode(); //memory_manager_.allocate();
        new_node->key = key;
        if (cur_size_ < max_size_) { // not full
            int value = cur_size_;
            new_node->value = value;
            cur_size_++;
            new_node->next = head_->next;
            head_->next->prev = new_node;
            new_node->prev = head_;
            head_->next = new_node;
            cache_[key] = new_node;
            return value;
        } else { // full
            LinkNode *free_node = tail_->prev;
            cache_.erase(free_node->key);
            free_node->prev->next = tail_;
            tail_->prev = free_node->prev;
            int value = free_node->value;
            new_node->next = head_->next;
            head_->next->prev = new_node;
            head_->next = new_node;
            new_node->prev = head_;
            new_node->value = value;
            cache_[key] = new_node;
            free_node->prev = nullptr;
            free_node->next = nullptr;
            //memory_manager_.free(free_node);
            delete free_node;
            return value;
        }
    } else {
        LinkNode *cur_node = it->second;
        cur_node->prev->next = cur_node->next;
        cur_node->next->prev = cur_node->prev;
        cur_node->next = head_->next;
        head_->next->prev = cur_node;
        head_->next = cur_node;
        cur_node->prev = head_;
        return cur_node->value;
    }
}
int LRU::View(int key) {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
        return -1;
    }
    LinkNode *cur_node = it->second;
    return cur_node->value;
}

int LRU::TryGet(int key) {
    auto it = cache_.find(key);
    if (it == cache_.end()) { // cannot find key
        LinkNode *new_node = new LinkNode(); //memory_manager_.allocate();
        new_node->key = key;
        if (cur_size_ < max_size_) { // not full
            int value = cur_size_;
            new_node->value = value;
            cur_size_++;
            new_node->next = head_->next;
            head_->next->prev = new_node;
            new_node->prev = head_;
            head_->next = new_node;
            cache_[key] = new_node;
            op_records_.emplace(OPType::Add, new_node->prev, new_node->next, new_node);
            return value;
        } else { // full
            LinkNode *free_node = tail_->prev;
            cache_.erase(free_node->key);
            free_node->prev->next = tail_;
            tail_->prev = free_node->prev;
            int value = free_node->value;
            new_node->next = head_->next;
            head_->next->prev = new_node;
            head_->next = new_node;
            new_node->prev = head_;
            new_node->value = value;
            cache_[key] = new_node;
            op_records_.emplace(OPType::Overflow, free_node->prev, free_node->next, free_node, free_node->key, free_node->value);
            return value;
        }
    } else {
        LinkNode *cur_node = it->second;
        op_records_.emplace(OPType::Get, cur_node->prev, cur_node->next);
        cur_node->prev->next = cur_node->next;
        cur_node->next->prev = cur_node->prev;
        cur_node->next = head_->next;
        head_->next->prev = cur_node;
        head_->next = cur_node;
        cur_node->prev = head_;
        return cur_node->value;
    }
}

void LRU::Rollback(int steps) {
    auto stack_len = op_records_.size();
    int min_step = stack_len > steps ? steps: stack_len;
    for (size_t i = 0; i < min_step; i++) {
        auto last_op = op_records_.top();
        op_records_.pop();
        switch (last_op.op_type) {
        case OPType::Add: {
            auto removed_node = last_op.removed_node;
            auto prev_node = last_op.prev_node;
            auto next_node = last_op.next_node;
            prev_node->next = next_node;
            next_node->prev = prev_node;
            removed_node->prev = nullptr;
            removed_node->next = nullptr;
            cache_.erase(removed_node->key);
            cur_size_--;
            // memory_manager_.free(removed_node);
            delete removed_node;
        }
            break;
        
        case OPType::Overflow: {
            auto new_node = head_->next;
            head_->next = new_node->next;
            new_node->next->prev = head_;
            new_node->next = nullptr;
            new_node->prev = nullptr;
            cache_.erase(new_node->key);
            delete new_node;
            // memory_manager_.free(new_node);
            auto removed_node = last_op.removed_node;
            auto prev_node = last_op.prev_node;
            auto next_node = last_op.next_node;
            prev_node->next = removed_node;
            removed_node->next = next_node;
            next_node->prev = removed_node;
            removed_node->prev = prev_node;
            int key = last_op.key;
            int value = last_op.value;
            removed_node->key = key;
            removed_node->value = value;
            cache_[key] = removed_node;
        }
            break;
        case OPType::Get: {
            auto cur_node = head_->next;
            head_->next = cur_node->next;
            cur_node->next->prev = head_;
            auto prev_node = last_op.prev_node;
            auto next_node = last_op.next_node;
            prev_node->next = cur_node;
            cur_node->next = next_node;
            next_node->prev = cur_node;
            cur_node->prev = prev_node;
        }
            break;
        }
    }   
}
void LRU::Show() {
    auto cur = head_->next;
    while (cur != tail_) {
        std::cout << "("<< cur->key << ", " << cur->value << "),";
        cur = cur->next;
    }
    //std::cout << std::endl;
}
std::vector<int> LRU::Keys() {
    auto cur = head_->next;
    std::vector<int> res;
    while (cur != tail_) {
        res.push_back(cur->key);
        cur = cur->next;
    }
    return res;
}
std::vector<int> LRU::Values() {
    auto cur = head_->next;
    std::vector<int> res;
    while (cur != tail_) {
        res.push_back(cur->value);
        cur = cur->next;
    }
    return res;
}
bool LRU::Contains(int key) {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
        return false;
    }
    return true;
}
py::list LRU::state_dict() {
    py::list ans;
    auto cur = head_->next;
    while (cur != tail_) {
        ans.append(py::make_tuple(cur->key, cur->value));
        cur = cur->next;
    }
    return ans;
}
void LRU::restore(pybind11::list kvs) {
    size_t len = kvs.size();
    if (len <= max_size_) {
        cur_size_ = 0;
        auto cur_node = head_;
        for (auto& kv : kvs) {
            auto new_node = new LinkNode(); //memory_manager_.allocate();
            new_node->key = kv.cast<py::tuple>()[0].cast<int>();
            new_node->value = kv.cast<py::tuple>()[1].cast<int>();
            cur_node->next = new_node;
            new_node->prev = cur_node;
            cache_[new_node->key] = new_node;
            cur_node = new_node;
            cur_size_++;
        }
        cur_node->next = tail_;
        tail_->prev = cur_node;
    }
}
