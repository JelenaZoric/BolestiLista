import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_adaptive, threshold_otsu
from skimage.morphology import closing, disk, dilation
from skimage.measure import label, regionprops
import csv
import random
import math
import operator
import cv2
# from sklearn import neighbors
file = open('podaci.csv', 'w')
# 'hwratio,eccentricity,solidity,species'
txt_to_write = []
brojVl = 13
brojKr = 13
for i in range(0, brojKr + brojVl):
    if i <= brojVl - 1:
        img = imread('slike/vl' + i.__str__() + '.jpg')
        vrsta = 0
    else:
        img = imread('slike/kr' + i.__str__() + '.jpg')
        vrsta = 1
# imgResized = img.resize((width, height))
    hsl, wsl = img.shape[:2]
#    plt.imshow(img)
# kmeans = KMeans(n_clusters=2).fit(img)
# kmeans.predict(img)
    imgGray = rgb2gray(img)
#    plt.imshow(imgGray, 'gray')
    thresh = threshold_otsu(imgGray)
    imgTreshold = imgGray <= thresh
# imgTreshold = 1 - threshold_adaptive(imgGray, block_size=75, offset=0.04)
#    plt.imshow(imgTreshold, 'gray')
    str_elem = disk(9)
    imgClosed = closing(imgTreshold, str_elem)
    imgClosed2 = closing(imgClosed, str_elem)
#    plt.imshow(imgClosed2, 'gray')
    labeledImg = label(imgClosed2)
    regions = regionprops(labeledImg)
    print('Regioni:{}'.format(len(regions)))
    regions_vineleaf = []
    regions_krompir = []


    def draw_regions(regs, img_size):
        img_res = np.ndarray((img_size[0], img_size[1]), dtype='float32')
        for reg in regs:
            coords = reg.coords
            for coord in coords:
                img_res[coord[0], coord[1]] = 1.
        return img_res


    for region in regions:
        bbox = region.bbox
        h = bbox[2] - bbox[0]
        w = bbox[3] - bbox[1]
        ratio = float(h) / w
        if ratio < 1.1:
            if h > hsl / 2 and w > wsl / 2:
                regions_vineleaf.append(region)
                newline = str(ratio) + ',' + str(region.eccentricity) + ',' + str(region.solidity) + ',' + str(vrsta)
                if i == 0:
                    txt_to_write.append(newline)
                else:
                    txt_to_write.append('\n' + newline)
                print('biljka je vinova loza')
#             plt.imshow(draw_regions(regions_vineleaf, imgClosed2.shape), 'gray')
        else:
            if h > hsl / 2:
                regions_krompir.append(region)
                newline = str(ratio) + ',' + str(region.eccentricity) + ',' + str(region.solidity) + ',' + str(vrsta)
                txt_to_write.append('\n' + newline)
                print('biljka je krompir')
#             plt.imshow(draw_regions(regions_psenica, imgClosed2.shape), 'gray')
# print(txt_to_write)
file.writelines(txt_to_write)
file.close()
#    plt.show()
def load_dataset(filename, split, training_set=[], test_set=[]):
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(3):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                training_set.append(dataset[x])
            else:
                test_set.append(dataset[x])


def euclidean_distance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def get_neighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1  # da ne uzima u obzir iza poslednje zapete, tj. naziv klase
    for x in range(len(trainingSet)):
        dist = euclidean_distance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def get_response(neighbors):  # vraca labele klasa najblizih suseda
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def get_accuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
       # for y in range(len(predictions[x])):
            if testSet[x][-1] == predictions[x]:
                correct += 1
               # break
    return (correct / float(len(testSet))) * 100.0


def main():
    trainingSet = []
    testSet = []
    split = 0.67
    load_dataset('podaci.csv', split, trainingSet, testSet)
    print 'Train set: ' + repr(len(trainingSet))  # broj training pdoataka
    print 'Test set: ' + repr(len(testSet))  # broj test podataka
    predictions = []
    predictionsForK = []
    k = 3
    for x in range(len(testSet)):
        neighbors = get_neighbors(trainingSet, testSet[x], k)
    #    result = []
        result = get_response(neighbors)
    #    predictions.append(result[0])
        predictionsForK.append(result)
        print('> predicted')
    #    for y in range(len(result)):
        print '>>'+repr(result)
        print('>>>>> actual=' + repr(testSet[x][-1]))
    accuracy = get_accuracy(testSet, predictionsForK)
    print('Accuracy for k nearest: ' + repr(accuracy) + '%')

    img = cv2.imread('slike/kr20.jpg')
    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    cv2.imshow('res2', res2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()
