import util


class DataPair:
    def __init__(self, point_x, point_y):
        self.point_x = point_x
        self.point_y = point_y
        self.distance = util.calculate_distance(point_x, point_y)



