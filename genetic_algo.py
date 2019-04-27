

class Customer:
    def __init__(self, x, y, distance, duration, id):
        self.x = x
        self.y = y
        self.distance = distance
        self.duration = duration
        self.id = id

    def __repr__(self):
        return 'D:{} ({:.3f},{:.3f})'.format(int(self.distance), self.x, self.y)

    def __lt__(self, other):
        return self.distance < other.distance

    def __cmp__(self, other):
        if hasattr(other, 'distance'):
            return self.distance.__cmp__(other.distance)