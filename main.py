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

class Camera:
    def __init__(self, ray, fov):
        self.ray = ray
        self.fov = fov

    def renderImage(self, scene, data, width, height):
        for x in range(width):
            for y in range(height):
                data[x,y] = [x,y,0]


class Color:
    def __init__(self, rgb):
        self.color = [rgb[0]/255, rgb[1]/255, rgb[2]/255]

    def restore(self):
        return [self.color[0]*255, self.color[1]*255, self.color[2]*255]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    width = 1024
    height = 1024
    data = np.zeros((width, height, 3), dtype = np.uint8)
    camera = Camera(Ray([0,0,0], [0,0,1]), np.pi/2)
    camera.renderImage(Scene(), data, width, height)

    image = Image.fromarray(data)
    image.show()