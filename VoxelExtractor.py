from typing import no_type_check_decorator
import numpy as np
from numpy.lib.arraysetops import unique
import pywavefront
import matplotlib.pyplot as plt
from PIL import Image

#Voxel Dictionary which stores all the voxels as voxel address/color pairs 'x:y:z' -> 'color'
voxelDict = {}
#Model array which stores the model and transformation matrix
modelArray = []
#Model dictionary which stores already loaded models so models don't have to be loaded multiple times
modelDictionary = {}
#Image dictionary which stores already loaded images so images don't have to be loaded multiple times
imageDictionary = {}

def getImage(textureName):
    if(textureName not in imageDictionary):
        image = Image.open("textures/"+textureName)
        imageDictionary[textureName] = image.convert('RGB')

    return imageDictionary[textureName]

class ColorSampler:
    def __init__(self, c, tex):
        self.color = c
        if tex != None:
            self.texture = getImage(tex.file_name)
        else:
            self.texture = None

    def sampleColor(self, u, v):
        if(self.texture != None):
            width, height = self.texture.size
            xPos = int(width * u)
            yPos = int(height * v)
            coordinate = min(xPos,width-1), min(yPos, height-1)
            colors = self.texture.getpixel(coordinate)
            return np.asarray(colors, dtype=float) / 255.0
        return self.color

def getModel(modelName):
    if(modelName not in modelDictionary):
        #Load the model
        model = pywavefront.Wavefront(modelName, collect_faces=True)
        modelDictionary[modelName] = model

    return modelDictionary[modelName]

class ModelEntry:
    def __init__(self, modelName, transform):
        self.model = getModel(modelName)
        self.transformationMatrix = transform

def getFaceArea(vertex1, vertex2, vertex3):
    v1v3Vec = vertex1 - vertex3
    v2v3Vec = vertex2 - vertex3
    crossProduct = np.cross(v1v3Vec, v2v3Vec)
    return np.sqrt(crossProduct.dot(crossProduct)) / 2

def sampleFace(vertex1, vertex2, vertex3, tex1, tex2, tex3, colorSampler, voxelScale):
    #Find the vertex vector with the greatest magnitude and create the ratio values
    v1Magnitude = np.sqrt(vertex1.dot(vertex1))
    v2Magnitude = np.sqrt(vertex2.dot(vertex2))
    v3Magnitude = np.sqrt(vertex3.dot(vertex3))
    if v1Magnitude > v2Magnitude and v1Magnitude > v3Magnitude:
        if(v2Magnitude > v3Magnitude):
            vectorList = [vertex1, vertex2, vertex3]
            texList = [tex1, tex2, tex3]
            ratio1 = v1Magnitude / v2Magnitude
        else:
            vectorList = [vertex1, vertex3, vertex2]
            texList = [tex1, tex3, tex1]
            ratio1 = v1Magnitude / v3Magnitude
        #ratio2 = v1Magnitude / v3Magnitude
    elif v2Magnitude > v3Magnitude:
        if(v1Magnitude > v3Magnitude):
            vectorList = [vertex2, vertex1, vertex3]
            texList = [tex2, tex1, tex3]
            ratio1 = v2Magnitude / v1Magnitude
        else:
            vectorList = [vertex2, vertex3, vertex1]
            texList = [tex2, tex3, tex1]
            ratio1 = v2Magnitude / v3Magnitude
        #ratio2 = v2Magnitude / v3Magnitude
    else:
        if v1Magnitude > v2Magnitude:
            vectorList = [vertex3, vertex1, vertex2]
            texList = [tex3, tex1, tex2]
            ratio1 = v3Magnitude / v1Magnitude
        else:
            vectorList = [vertex3, vertex2, vertex1]
            texList = [tex3, tex2, tex1]
            ratio1 = v3Magnitude / v2Magnitude
        #ratio2 = v3Magnitude / v2Magnitude
    
    #Initialize barycentric coordinates
    u = 0.0
    v = 0.0
    w = 0.0
    #Use the area of the rectangle to get the area which is used for the sqrt function
    area = getFaceArea(vertex1, vertex2, vertex3) * 2
    numSamples = int(np.sqrt(area)*ratio1 * 1.25 * voxelScale)
    numSamples = max(numSamples, 3)

    points = []
    #Evenly sample points along the triangle
    for i in range(numSamples + 1):
        u = i * (1.0 / numSamples)
        for j in range(int(numSamples / ratio1) + 1):
            v = j * (1.0 / numSamples) * ratio1
            #for k in range(int(numSamples / ratio2) + 1):
            w = 1.0 - u - v
            #Ensure the point is in barycentric coordiantes (u+v+w == 1 and all coordinates are non-zero with a slight buffer)
            if(abs(1.0 - (u + v + w)) < 0.0001 and w >= -0.0001):
                #Caclulate the point based on the barycentric coordinates
                point = u * vectorList[0] + v * vectorList[1] + w * vectorList[2]
                voxelPoint = np.round(point * voxelScale)
                voxelKey = f'{round(voxelPoint[0])}:{round(voxelPoint[1])}:{round(voxelPoint[2])}'
                texU = texList[0][0] * u + texList[1][0] * v + texList[2][0] * w
                texV = texList[0][1] * u + texList[1][1] * v + texList[2][1] * w
                if(voxelKey in voxelDict):
                    #Average the colors
                    voxelDict[voxelKey] = (voxelDict[voxelKey] + colorSampler.sampleColor(u, v)) / 2
                else:
                    #Add the key to the dict
                    voxelDict[voxelKey] = colorSampler.sampleColor(texU, texV)

def writeVoxelFile():
    with open("scene.vox", "w") as output:
        for key, value in voxelDict.items():
            voxelCoords = key.split(":")
            red = int(value[0] * 255)
            green = int(value[1] * 255)
            blue = int(value[2] * 255)
            color = (red << 16) | (green << 8) | (blue)
            output.write(f'{voxelCoords[0]},{voxelCoords[1]},{voxelCoords[2]},{color}\n')
    print("Finished Writing to File!")

def createTranslationMatrix(x, y, z):
    matrix = np.identity(4, dtype=float)
    matrix[3][0] = x
    matrix[3][1] = y
    matrix[3][2] = z
    return matrix

def createScaleMatrix(xScale, yScale, zScale):
    matrix = np.identity(4, dtype=float)
    matrix[0][0] = xScale
    matrix[1][1] = yScale
    matrix[2][2] = zScale
    return matrix

def readVoxelScene():
    with open("sceneDesc.txt", "r") as input:
        lines = input.readlines()
        for line in lines:
            #Skip lines that start will comments
            if line.startswith("//"):
                continue
            components = line.split(' ')
            #Ensure the length of the line is greator than 1
            if len(components) > 1:
                modelName = components[0]
                #Get the translation information
                transform = np.identity(4, dtype=float)
                translationCoords = components[1].split(',')
                #Get the scaling information
                scale = np.identity(4, dtype=float)
                if len(components) > 2:
                    scaleComponents = components[2].split(',')
                    scale = createScaleMatrix(scaleComponents[0], scaleComponents[1], scaleComponents[2])
                transform = np.matmul(transform, scale)
                transform = np.matmul(transform, createTranslationMatrix(float(translationCoords[0]), float(translationCoords[1]), float(translationCoords[2])))
                modelArray.append(ModelEntry(modelName, transform))

def main():
    #Scale of the voxels. I.e. 2 means there are two voxels for a single unit (0.0-1.0). 4 means there are four voxels within a single unit (0.0-1.0)
    voxelScale = int(input("Enter the Voxel Scale: "))

    readVoxelScene()
    for modelEntry in modelArray:
        for name, material in modelEntry.model.materials.items():
            #Process a face
            offset = 0
            if material.has_normals:
                offset += 3
            if material.has_uvs:
                offset += 2
            total_faces = (len(material.vertices) / material.vertex_size) / 3
            face_size = material.vertex_size * 3

            for i in range(int(total_faces)):
                tex1 = tex2 = tex3 = [0,0]

                vertex1X = material.vertices[i * face_size + offset]
                vertex1Y = material.vertices[i * face_size + offset + 1]
                vertex1Z = material.vertices[i * face_size + offset + 2]
                vertex1 = np.array([vertex1X, vertex1Y, vertex1Z, 1.0])
                vertex1 = vertex1.dot(modelEntry.transformationMatrix)
                vertex1 = np.delete(vertex1, 3, 0)

                vertex2X = material.vertices[i * face_size + material.vertex_size + offset]
                vertex2Y = material.vertices[i * face_size + material.vertex_size + offset + 1]
                vertex2Z = material.vertices[i * face_size + material.vertex_size + offset + 2]
                vertex2 = np.array([vertex2X, vertex2Y, vertex2Z, 1.0])
                vertex2 = vertex2.dot(modelEntry.transformationMatrix)
                vertex2 = np.delete(vertex2, 3, 0)

                vertex3X = material.vertices[i * face_size + material.vertex_size * 2 + offset]
                vertex3Y = material.vertices[i * face_size + material.vertex_size * 2 + offset + 1]
                vertex3Z = material.vertices[i * face_size + material.vertex_size * 2 + offset + 2]
                vertex3 = np.array([vertex3X, vertex3Y, vertex3Z, 1.0])
                vertex3 = vertex3.dot(modelEntry.transformationMatrix)
                vertex3 = np.delete(vertex3, 3, 0)

                if(material.texture == None):
                    tex1 = [material.vertices[i * face_size + offset - 2], 
                        material.vertices[i * face_size + offset - 2 + 1]]
                    tex2 = [material.vertices[i * face_size + material.vertex_size + offset - 2],
                        material.vertices[i * face_size + material.vertex_size + offset - 2 + 1]]
                    tex3 = [material.vertices[i * face_size + material.vertex_size * 2 + offset - 2],
                        material.vertices[i * face_size + material.vertex_size * 2 + offset - 2 + 1]]
                    colorSampler = ColorSampler(np.array([material.diffuse[0], material.diffuse[1], material.diffuse[2]]), None)
                else:
                    colorSampler = ColorSampler(np.array([0, 0, 0]), material.texture)
                sampleFace(vertex1, vertex2, vertex3, tex1, tex2, tex3, colorSampler, voxelScale)
                
    writeVoxelFile()

if __name__ == "__main__":
    main()