from main import util


class DataPair:
    def __init__(self, point_x, point_y):
        self.point_x = point_x
        self.point_y = point_y

        # assert (is_x_positive != is_y_positive)
        #
        # self.is_x_positive = is_x_positive
        # self.is_y_positive = is_y_positive
        self.distance = util.calculate_distance(point_x, point_y)

    def calculate_mid_point(self):
        mid_point = []
        for i in range(len(self.point_x)):
            val = (self.point_x[i] + self.point_y[i]) / 2
            mid_point.append(val)

        return mid_point
