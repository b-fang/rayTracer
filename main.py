from PIL import Image
import numpy as np

# global approximation value
EPSILON = 0.01
N = 6

class Scene:
    def __init__(self):
        self.geometry = [
            Sphere([1.1, 0, 3], 1, Material(1, [100, 200, 100])),
            Sphere([-1.1, 0, 3], 1, Material(0.4, [200, 100, 100])),
            Plane([0, 1, 0], [0, -1, 0], Material(1, [100, 100, 200]))
        ]

    def storeGeometry(self):
        pass

    def shootRay(self, ray, depth=0):
        if depth > 3:
            return Color([0, 0, 0])
        intersection = None
        closestShape = None
        for shape in self.geometry:
            tempIntersection = shape.getIntersection(ray)
            if tempIntersection is not None and self.closer(tempIntersection, intersection, ray.origin):
                intersection = tempIntersection
                closestShape = shape
        if intersection is not None:
            return closestShape.computeLight(ray, intersection, self, depth)
        brightness = (ray.direction[1]+1)/2
        return Color([255, 255, 255]).scale(1-brightness).add(Color([120, 190, 255]).scale(brightness))

    def closer(self, tempIntersection, intersection, origin):
        if intersection is None:
            return True
        v1 = tempIntersection - origin
        v2 = intersection - origin
        return np.dot(v1, v1) < np.dot(v2, v2)


class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin)
        self.direction = np.array(direction)/np.linalg.norm(direction)


class Material:
    def __init__(self, diffusivity, color):
        self.diffusivity = diffusivity
        self.color = Color(color)

    def getDirections(self, ray, normal, scene):
        directions = []
        direction = ray.direction-2*normal*np.dot(ray.direction, normal)
        center = (1-self.diffusivity)*direction + self.diffusivity*normal
        r = self.diffusivity
        n = 1
        for i in range(n):
            directions.append(self.diffusivity*self.getRandSpherePoint()+center)
        return directions

    def getRandSpherePoint(self):
        z = np.random.random()*2-1
        r = np.sqrt(1-z**2)
        theta = np.random.random()*2*np.pi
        return np.array([r*np.cos(theta), r*np.sin(theta), z])


class Shape:
    def computeLight(self, ray, intersection, scene, depth):
        directions = self.material.getDirections(ray, self.getNormal(intersection), scene)
        color = Color([0, 0, 0])
        for direction in directions:
            color2 = scene.shootRay(Ray(intersection+EPSILON*direction, direction), depth+1)
            color2 = color2.scale(1/len(directions))
            color = color.add(color2)
        return color.multiply(self.material.color)


class Plane(Shape):
    def __init__(self, normal, point, material):
        self.normal = np.array(normal)
        self.point = np.array(point)
        self.material = material

    def getIntersection(self, ray):
        # make sure that dot product of outgoing ray and plane is not extremely small/zero
        denom = np.dot(ray.direction, self.normal)
        if np.abs(denom) < EPSILON:
            return None
        t = np.dot((self.point-ray.origin), self.normal)/denom
        if t >= 0:
            return ray.origin + t*ray.direction
        return None

    def getNormal(self, intersection):
        return self.normal


class Sphere(Shape):
    def __init__(self, origin, radius, material):
        self.origin = np.array(origin)
        self.r = radius
        self.material = material

    def getIntersection(self, ray):
        t = (np.dot(self.origin, ray.direction) - np.dot(ray.origin, ray.direction))/np.dot(ray.direction, ray.direction)
        v = ray.origin + t*ray.direction
        vq = self.origin-v
        if np.dot(vq, vq) < self.r**2 and t >= 0:
            y = np.sqrt(self.r**2-np.dot(vq, vq))
            return ray.origin+(t-y)*ray.direction
        return None

    def getNormal(self, intersection):
        return (intersection-self.origin)/self.r


class Light:
    def __init__(self, position, color):
        self.position = position
        self.color = Color(color)


class Camera:
    def __init__(self, ray, fov):
        self.ray = ray
        self.fov = fov
        d = self.ray.direction
        j = np.array([0, 1, 0])
        right = np.cross(d, j)
        right = right/np.linalg.norm(right)
        up = np.cross(right, d)
        up = up/np.linalg.norm(up)
        # np.column_stack arranges the vectors as columns and creates a matrix
        self.matrix = np.column_stack((right, up, d))

    def renderImage(self, scene, data, width, height):
        tan = np.tan(self.fov/2)
        for x in range(width):
            nx = 2*x/width - 1
            print(x)
            for y in range(height):
                ny = 1 - 2*y/height
                a = width/height
                direction = np.array([a*tan*nx, tan*ny, 1])
                # np.matmul returns the matrix from the matrix multiplication
                colorTemp = Color()
                for i in range(N):
                    normalizedRandomX = np.random.random()*2 - 1
                    normalizedRandomY = np.random.random()*2 - 1
                    adjustment = np.array([normalizedRandomX*a*tan/width, normalizedRandomY*tan/height, 0])
                    directionTransformed = np.matmul(self.matrix, direction + adjustment)
                    colorTemp = colorTemp.add(scene.shootRay(Ray(self.ray.origin, directionTransformed)).scale(1/N))
                data[y, x] = colorTemp.restore()


class Color:
    def __init__(self, rgb=[0, 0, 0]):
        self.color = np.array([rgb[0]/255, rgb[1]/255, rgb[2]/255])

    def restore(self):
        return [np.floor(self.color[0]*255), np.floor(self.color[1]*255), np.floor(self.color[2]*255)]

    def add(self, color):
        newColor = np.clip(self.color+color.color, 0, 1)
        return Color(newColor*255)

    def scale(self, scalar):
        return Color(self.color*scalar*255)

    def multiply(self, color):
        return Color(np.multiply(self.color, color.color)*255)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    width = 256
    height = 256
    data = np.zeros((height, width, 3), dtype=np.uint8)
    camera = Camera(Ray([0, -0.5, 0], [0, 0.20, 1]), np.pi/2)
    camera.renderImage(Scene(), data, width, height)

    image = Image.fromarray(data)
    image.save("output/sphere.png")
    image.show()
