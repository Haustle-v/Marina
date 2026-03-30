import cloudpickle


def ab_sum(a, b):
    return a + b


def square(x):
    return x * x

class Counter(object):
    def __init__(self):
        self.n = 0

    def increment(self):
        self.n += 1

    def read(self):
        return self.n

with open("cloudpickle_ab_sum.pkl", "wb") as f:
    cloudpickle.dump(ab_sum, f)

with open("cloudpickle_counter.pkl", "wb") as f:
    cloudpickle.dump(Counter, f)
