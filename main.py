from PIL import Image
import numpy as np


class Scene:
    def storeGeometry(self):
        pass

    def shootRay(self):
        pass


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin;
        self.direction = direction;


class Plane:
    def getIntersection(self, ray):
        pass

    def computeLight(self, ray):
        pass


class Sphere:
    def getIntersection(self, ray):
        pass

    def computeLight(self, ray):
        pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = np.zeros((1024, 1024, 3), dtype = np.uint8)

    for i in range(100):
        for j in range(100):
            data[i, j] = [255, 0, 0]

    data[100, 101] = [0, 255, 0]
    data[101, 100] = [0, 0, 255]

    image = Image.fromarray(data)
    image.show()