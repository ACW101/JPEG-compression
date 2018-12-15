from collections import OrderedDict


class LRU:
    def __init__(self, capacity):
        self.dic = OrderedDict()
        self.vacancy = capacity
        self.next_val = 256

    def get(self, key):
        if key not in self.dic:
            return None
        self.dic.move_to_end(key, last=False)  # move to the front
        return self.dic[key]

    def set(self, key):
        if key in self.dic:
            self.dic.move_to_end(key, last=False)  # move to the front
        else:
            value = self.next_val
            if self.vacancy > 0:
                self.vacancy -= 1
                self.next_val += 1
            else:
                (key, val) = self.dic.popitem(last=True)  # pop the last element
                # reuse popped value
                value = val
        self.dic[key] = value  # set new val
        self.dic.move_to_end(key, last=False)  # move to the front

    def __iter__(self):
        return self.dic.__iter__()
