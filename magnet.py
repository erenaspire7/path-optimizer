import random


class Magnet:
    def __init__(self, x, y, curve_point):
        self.x = x
        self.y = y
        self.curve_point = curve_point

        if curve_point:
            self.magnetic_strength = random.uniform(0, 255)
        else:
            self.magnetic_strength = 255

        self.active = False
        self.start_time = 0
        self.end_time = 0

    def to_tuple(self):
        return (self.x, self.y)

    def activate(self, time):
        self.active = True
        self.start_time = time

    def deactivate(self, time):
        self.active = False
        self.end_time = time

    def to_dict(self):
        return {
            "x": self.x.item(),
            "y": self.y.item(),
            "magnetic_strength": self.magnetic_strength,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }
