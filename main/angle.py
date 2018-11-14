from main import util


class Angle:
    def __init__(self, center, point1, point2):
        self.center = center
        self.point1 = point1
        self.point2 = point2
        self.vector1, self.vector2, self.angle = self.compute_angle()

    def compute_angle(self):
        dim = len(self.center)
        vector1 = []
        vector2 = []
        for i in range(dim):
            vector1.append(self.point1[i] - self.center[i])
            vector2.append(self.point2[i] - self.center[i])

        angle = util.calculate_vector_angle(vector1, vector2)
        return vector1, vector2, angle

    def find(self, point_x, point_y):
        if self.point1 == point_x and self.point2 == point_y:
            return True
        if self.point1 == point_y and self.point2 == point_x:
            return True

        return False

    def get_other_point(self, point):
        if self.point1 == point:
            return self.point2
        elif self.point2 == point:
            return self.point1
        else:
            return None
