import argparse
import sys
from bitarray import bitarray
from LRU import LRU
dictionary_size = 64000


def create_initial_dict(isLru=False):
    if isLru:
        return LRU(dictionary_size - 256)
    return {}


def create_initial_lookup_arr():
    return [chr(i) for i in range(256)]


class Decoder:
    def __init__(self, input, update="freeze"):
        if update == "LRU":
            self.local_dict = create_initial_dict(isLru=True)
        else:
            self.local_dict = create_initial_dict()
            self.lookup_arr = create_initial_lookup_arr()
        self.static_dict = {i: chr(i) for i in range(256)}
        self.dict_count = 256
        self.update = update
        self.input = input

    def fc(self):
        prev_match = ""
        byte_arr = bytearray(self.input.read())
        i = 0
        res = bitarray(endian="big")
        while i < len(byte_arr):
            code = int.from_bytes(byte_arr[i:i+2], "big")
            i += 2
            cur_match = ""
            # range latin-1
            if code < 256:
                cur_match = self.static_dict[code]
                res += bitarray(cur_match, endian="big")
            elif self.update == "LRU":
                for (key, val) in self.local_dict.dic.items():
                    if val == code:
                        cur_match = key
                        self.local_dict.get(cur_match)
                        res += bitarray(cur_match, endian="big")
                        break
            else:
                cur_match = self.lookup_arr[code]
                res += bitarray(cur_match, endian="big")

            if self.update != "freeze" or self.dict_count < dictionary_size:
                prev_plus_FC = prev_match + cur_match[0]
                if len(prev_plus_FC) > 1 and prev_plus_FC not in self.local_dict:
                    if self.update == "LRU":
                        self.local_dict.set(prev_plus_FC)
                    else:
                        self.local_dict[prev_plus_FC] = self.dict_count
                        self.lookup_arr.append(prev_plus_FC)
                    self.dict_count += 1
                    # refresh logic
                    if self.update == "restart" and self.dict_count == dictionary_size:
                        print("restart dict")
                        self.__init__(None)
            prev_match = cur_match
        sys.stdout.buffer.write(res.tobytes())

    def ap(self):
        prev_match = ""
        byte_arr = bytearray(self.input.read())
        i = 0
        res = bitarray(endian="big")
        while i < len(byte_arr):
            code = int.from_bytes(byte_arr[i:i+2], "big")
            i += 2
            cur_match = ""
            if code < 256:
                cur_match = self.static_dict[code]
                res += bitarray(cur_match, endian="big")
            elif self.update == "LRU":
                for (key, val) in self.local_dict.dic.items():
                    if val == code:
                        cur_match = key
                        self.local_dict.get(cur_match)
                        res += bitarray(cur_match, endian="big")
                        break
            else:
                cur_match = self.lookup_arr[code]
                res += bitarray(cur_match, endian="big")

            if self.update != "freeze" or self.dict_count < dictionary_size:
                all_prefix = [cur_match[0:i]
                              for i in range(1, len(cur_match) + 1)]
                for prefix in all_prefix:
                    prev_plus_prefix = prev_match + prefix
                    if len(prev_plus_prefix) > 1 and prev_plus_prefix not in self.local_dict:
                        prev_plus_prefix = prev_match + prefix
                        if self.update == "LRU":
                            self.local_dict.set(prev_plus_prefix)
                        else:
                            self.local_dict[prev_plus_prefix] = self.dict_count
                            self.lookup_arr.append(prev_match + prefix)
                        self.dict_count += 1
                        # refresh logic
                        if self.update == "restart" and self.dict_count == dictionary_size:
                            print("restart dict")
                            self.__init__(None)
            prev_match = cur_match
        sys.stdout.buffer.write(res.tobytes())


if __name__ == "__main__":
    decoder = Decoder(sys.stdin.buffer, update="restart")
    decoder.fc()
