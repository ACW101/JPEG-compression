dictionary_size = 64000
import argparse
import sys
from LRU import LRU
from bitarray import bitarray


def create_initial_dict(isLru=False):
    if isLru:
        return LRU(dictionary_size - 256)
    return {}


class Encoder:
    def __init__(self, input, update="freeze"):
        if update == "LRU":
            self.local_dict = create_initial_dict(isLru=True)
        else:
            self.local_dict = create_initial_dict()
        self.dict_count = 256
        self.static_dict = {chr(i): i for i in range(256)}
        self.update = update
        self.input = input

    def fc(self):
        f = self.input
        out = sys.stdout.buffer
        text = bitarray(endian="big")
        text.frombytes(f.read())
        text_len = len(text)
        prev_match = ""
        cur_match = text[0]
        i = 0
        while i < text_len:
            # get longest cur_match possible
            test_cur_end = i + 2
            while test_cur_end <= text_len and text[i:test_cur_end].to01() in self.local_dict:
                test_cur_end += 1
            cur_match = text[i:test_cur_end - 1].to01()
            i = test_cur_end - 1

            # latin-1 range
            if len(cur_match) == 1:
                code = self.static_dict[cur_match]
                out.write(code.to_bytes(2, "big"))
            # write binary to file
            elif self.update == "LRU":
                code = self.local_dict.get(cur_match)
                out.write(code.to_bytes(2, "big"))
            else:
                code = self.local_dict[cur_match]
                out.write(code.to_bytes(2, "big"))

            # update dictionary
            if self.update != "freeze" or self.dict_count < dictionary_size:
                prev_plus_FC = prev_match + cur_match[0]
                if len(prev_plus_FC) > 1 and prev_plus_FC not in self.local_dict:
                    if self.update == "LRU":
                        self.local_dict.set(prev_plus_FC)
                    else:
                        self.local_dict[prev_plus_FC] = self.dict_count
                    self.dict_count += 1
                    # restart dictionary if full
                    if self.update == "restart" and self.dict_count == dictionary_size:
                        self.dict_count = 256
                        self.local_dict = create_initial_dict()
            prev_match = cur_match

    def ap(self):
        f = self.input
        out = sys.stdout.buffer
        text = bitarray(endian="big")
        text.frombytes(f.read())
        text_len = len(text)
        prev_match = ""
        cur_match = text[0]
        i = 0
        while i < text_len:
                # get longest cur_match possible
            test_cur_end = i + 2
            while test_cur_end <= text_len and text[i:test_cur_end].to01() in self.local_dict:
                test_cur_end += 1
            cur_match = text[i:test_cur_end - 1].to01()
            i = test_cur_end - 1

            if len(cur_match) == 1:
                code = self.static_dict[cur_match]
                out.write(code.to_bytes(2, "big"))
            # write binary to file
            elif self.update == "LRU":
                code = self.local_dict.get(cur_match)
                out.write(code.to_bytes(2, "big"))
            else:
                code = self.local_dict[cur_match]
                out.write(code.to_bytes(2, "big"))

            # update dictionary
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
                        self.dict_count += 1
                        # restart logic
                        if self.update == "restart" and self.dict_count == dictionary_size:
                            self.dict_count = 256
                            self.local_dict = create_initial_dict()
            prev_match = cur_match


if __name__ == "__main__":
    encoder = Encoder(sys.stdin.buffer, update="restart")
    encoder.fc()
