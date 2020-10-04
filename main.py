from PIL import Image
import numpy as np


class Scene:
    def __init__(self):
        self.sphere = Sphere([0,0,5], 1)

    def storeGeometry(self):
        pass

    def shootRay(self, ray):
        if self.sphere.getIntersection(ray) is not None:
            return Color([255,0,0])#Color([255*np.dot(ray.direction, [0,0,1]), 0, 0])
        return Color([255,255,255])


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin;
        self.direction = direction/np.linalg.norm(direction)


class Plane:
    def getIntersection(self, ray):
        pass

    def computeLight(self, ray):
        pass


class Sphere:
    def __init__(self, origin, radius):
        self.origin = origin
        self.r = radius

    def getIntersection(self, ray):
        t = (np.dot(self.origin, ray.direction)- np.dot(ray.origin, ray.direction))/np.dot(ray.direction, ray.direction)
        v = ray.origin + t*ray.direction
        vq = self.origin-v
        if np.dot(vq,vq) < self.r**2:
            y = np.sqrt(self.r**2-np.dot(vq, vq))
            return ray.origin+(t-y)*ray.direction
        return None

    def computeLight(self, ray):
        pass

class Camera:
    def __init__(self, ray, fov):
        self.ray = ray
        self.fov = fov

    def renderImage(self, scene, data, width, height):
        for x in range(width):
            nx = 2*x/width - 1
            for y in range(height):
                ny = 1 - 2*y/height
                a = width/height
                direction = [a*np.tan(self.fov/2)*nx, np.tan(self.fov/2)*ny, 1]
                data[x,y] = scene.shootRay(Ray(self.ray.origin, direction)).restore()


class Color:
    def __init__(self, rgb):
        self.color = [rgb[0]/255, rgb[1]/255, rgb[2]/255]

    def restore(self):
        return [self.color[0]*255, self.color[1]*255, self.color[2]*255]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    width = 512
    height = 512
    data = np.zeros((width, height, 3), dtype = np.uint8)
    camera = Camera(Ray([0,0,0], [0,0,1]), np.pi/2)
    camera.renderImage(Scene(), data, width, height)



    image = Image.fromarray(data)
    image.show()