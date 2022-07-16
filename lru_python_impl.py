from _weakref import proxy as _proxy


class Op(object):
    def __init__(self, op_type, prev_node=None, next_node=None, removed_node=None, key=None, value=None):
        self.op_type = op_type
        self.prev_node = prev_node
        self.next_node = next_node
        self.removed_node = removed_node
        self.kv = (key, value)


class LinkNode(object):
    __slots__ = 'prev', 'next', 'key', 'value', '__weakref__'


class LRU:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cur_idx = 0
        self.cache = {}  # map key to the double link list.
        self.__hard_head = LinkNode()
        self.__hard_tail = LinkNode()
        self.__head = _proxy(self.__hard_head)
        self.__head.key = '___head___'
        self.__tail = _proxy(self.__hard_tail)
        self.__tail.key = '___tail___'
        self.__head.next = self.__tail
        self.__tail.prev = self.__head
        self.__head.prev = self.__tail.next = None
        self.op_stack = []

    def get(self, key):
        if key in self.cache:
            cur_node = self.cache[key]
            value = cur_node.value
            prev_node, next_node = cur_node.prev, cur_node.next
            next_node.prev = prev_node
            prev_node.next = next_node
            # insert head
            cur_node.next = self.__head.next
            self.__head.next.prev = cur_node
            cur_node.prev = self.__head
            self.__head.next = cur_node
            return value
        else:
            new_node = LinkNode()
            new_node.key = key
            if self.cur_idx < self.capacity:  # not full
                r = self.cur_idx
                self.cache[key] = new_node
                new_node.value = r
                self.cur_idx += 1
                new_node.next = self.__head.next
                self.__head.next.prev = new_node
                new_node.prev = self.__head
                self.__head.next = new_node
                return r
            else:
                free_node = self.__tail.prev
                free_node.prev.next = self.__tail
                self.__tail.prev = free_node.prev
                old_key = free_node.key
                r = free_node.value
                free_node.prev = free_node.next = None
                self.cache.pop(old_key)
                new_node.next = self.__head.next
                self.__head.next.prev = new_node
                self.__head.next = new_node
                new_node.prev = self.__head
                new_node.value = r
                self.cache[key] = new_node
                return r

    def __iter__(self):
        cur = self.__head.next
        while cur.key != self.__tail.key:
            # print(cur.key)
            yield cur.key, cur.value
            cur = cur.next

    def state_dict(self):
        cur = self.__head.next
        kvs = []
        while cur.key != self.__tail.key:
            # print(cur.key)
            kvs.append((cur.key, cur.value))
            cur = cur.next
        return kvs

    def restore(self, kvs):
        assert len(kvs) <= self.capacity
        assert self.cur_idx == 0
        cur_node = self.__head
        for kv in kvs:
            assert kv[0] not in self.cache
            new_node = LinkNode()
            new_node.key = kv[0]
            new_node.value = kv[1]
            self.cache[kv[0]] = new_node
            cur_node.next = new_node
            new_node.prev = cur_node
            cur_node = new_node
            self.cur_idx += 1
        cur_node.next = self.__tail
        self.__tail.prev = cur_node


    def clear(self):
        self.cache.clear()
        self.__head.next = self.__tail
        self.__tail.prev = self.__head

    def __contains__(self, key):
        return key in self.cache

    def view(self, key):
        if key not in self.cache:
            return -1
        else:
            return self.cache[key].value

    def keys(self):
        return self.cache.keys()

    def try_get(self, key):
        if key in self.cache:
            cur_node = self.cache[key]
            value = cur_node.value
            prev_node, next_node = cur_node.prev, cur_node.next
            op_records = Op('Get', prev_node, next_node)
            self.op_stack.append(op_records)
            next_node.prev = prev_node
            prev_node.next = next_node
            # insert head
            cur_node.next = self.__head.next
            self.__head.next.prev = cur_node
            cur_node.prev = self.__head
            self.__head.next = cur_node
            return value
        else:
            new_node = LinkNode()
            new_node.key = key
            if self.cur_idx < self.capacity:  # not full
                r = self.cur_idx
                self.cache[key] = new_node
                new_node.value = r
                self.cur_idx += 1
                # insert head
                new_node.next = self.__head.next
                self.__head.next.prev = new_node
                new_node.prev = self.__head
                self.__head.next = new_node
                op_records = Op('Add', new_node.prev, new_node.next, new_node)
                self.op_stack.append(op_records)
                return r
            else:
                free_node = self.__tail.prev
                free_node.prev.next = self.__tail
                self.__tail.prev = free_node.prev
                old_key = free_node.key
                r = free_node.value
                op_records = Op(
                    'Overflow', free_node.prev, free_node.next, free_node, key=old_key, value=free_node.value
                )
                self.op_stack.append(op_records)
                free_node.prev = free_node.next = None
                self.cache.pop(old_key)
                new_node.next = self.__head.next
                self.__head.next.prev = new_node
                self.__head.next = new_node
                new_node.prev = self.__head
                new_node.value = r
                self.cache[key] = new_node
                return r

    def rollback_one_step(self, ):
        if len(self.op_stack) > 0:
            last_op = self.op_stack.pop()
            if last_op.op_type == 'Add':
                removed_node = last_op.removed_node
                prev_node, next_node = last_op.prev_node, last_op.next_node
                prev_node.next = next_node
                next_node.prev = prev_node
                removed_node.prev = removed_node.next = None
                self.cache.pop(removed_node.key)
                self.cur_idx -= 1
            elif last_op.op_type == 'Overflow':
                # first remove new node
                new_node = self.__head.next
                self.__head.next = new_node.next
                new_node.next.prev = self.__head
                new_node.next = new_node.prev = None
                self.cache.pop(new_node.key)
                # add old node
                removed_node = last_op.removed_node
                prev_node = last_op.prev_node
                next_node = last_op.next_node
                prev_node.next = removed_node
                removed_node.prev = prev_node
                removed_node.next = next_node
                next_node.prev = removed_node
                key = last_op.kv[0]
                removed_node.key = key
                removed_node.value = last_op.kv[1]
                assert key not in self.cache
                self.cache[key] = removed_node
            else:
                assert last_op.op_type == 'Get'
                cur_node = self.__head.next
                self.__head.next = cur_node.next
                cur_node.next.prev = self.__head
                last_op.prev_node.next = cur_node
                cur_node.prev = last_op.prev_node
                cur_node.next = last_op.next_node
                last_op.next_node.prev = cur_node

    def rollback_steps(self, steps):
        max_steps = min(steps, len(self.op_stack))
        for _ in range(max_steps):
            self.rollback_one_step()
