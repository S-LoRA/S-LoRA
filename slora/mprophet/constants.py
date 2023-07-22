GB = 1024 ** 3
T = 1000 ** 4


def get_num_bytes(dtype):
    if dtype == "fp16":
        return 2
    else:
        raise NotImplementedError

