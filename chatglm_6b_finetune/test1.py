import math
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

    def __post_init__(self):
        self.distance_from_origin = math.sqrt(self.x ** 2 + self.y ** 2)

    def distance_to(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

p1 = Point(3, 4)
p2 = Point(6, 8)

# print(p1.distance_from_origin)  # 输出 5.0
print(p1.distance_to(p2))  # 输出 5.0
