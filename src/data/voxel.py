class Voxel:
    def __init__(self, _unused):
        self.sum = 0
        self.num = 0

    def append(self, val):
        self.sum += val
        self.num+= 1

    def mean(self):
        if self.num > 0:
            return self.sum / self.num
        else:
            return 0
