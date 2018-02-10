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
from sklearn.cluster import KMeans
# from sklearn import neighbors
file = open('podaci.csv', 'w')
# 'hwratio,eccentricity,solidity,species'
txt_to_write = []
brojVl = 13
brojKr = 13
fileBolesti = open('bolesti.csv', 'r')
brojTacnih = 0
for i in range(0, brojKr + brojVl):
    if i <= brojVl - 1:
        image = cv2.imread('slike/vl' + i.__str__() + '.jpg')
    else:
        image = cv2.imread('slike/kr' + i.__str__() + '.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.reshape((image.shape[0] * image.shape[1], 3))

    clt = KMeans(n_clusters=3)
    clt.fit(image)

    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, clt.cluster_centers_):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)
        startX = endX

    # plt.figure()
    # plt.axis("off")
    # plt.imshow(bar)
    # print(bar)
    # l = chain.from_iterable(zip(*bar))
    # list(l)

    # print(list(l))
    width, height = bar.shape[:2]
    pixel_values = list(bar)
    pixel_values = np.array(pixel_values).reshape((width, height, 3))  # w = 50, h = 300

    bolest = 0
    predvidjena_bolest = fileBolesti.readline()
    predvidjena_bolest = float(predvidjena_bolest)
    # print(pixel_values[1][1][2])  #u poslednjoj zagradi se kaze da li je r g ili b
    # # pix = bar.load()
    # for x in range(width):
    for y in range(height):
        # for z in range(3):

        if pixel_values[1][y][2] > 70 and pixel_values[1][y][2] < 200:
            # print(pixel_values[1][y][2])
            bolest = 1
            print('Biljka ' + i.__str__() + ' je bolesna.')
            break
    if bolest == 0:
        print('Biljka ' + i.__str__() + ' nije bolesna.')
    if bolest == predvidjena_bolest:
        brojTacnih += 1
        print('Pretpostavka bolesti je tacna.')
    else:
        print('Pretpostavka bolesti je netacna.')
tacnost = float(brojTacnih)/(brojVl+brojKr)*100
print('Procenat tacnosti prepoznavanja bolesti je: ' + str(tacnost) + '%')
print('Preporuke za lecenje plamenjace: ')
print('\t- Preparat Polyram DF (BASF):')
print('\t\tKolicina primene: 0,2% koncentracije za vinovu lozu, odnosno 1,5-2 kg/ha za krompir')
print('\t\tVreme primene: pre ostvarivanja uslova za sekundarne infekcije')
print('\t\tKarenca: za stone sorte vinove loze 28 dana, za vinske sorte vinove loze 35 dana, a za krompir 14 dana')
print('\t- Preparat Acrobat MZ WG (BASF):')
print('\t\tKolicina primene: 2,0-2,5 kg/ha za vinovu lozu i krompir')
print('\t\tVreme primene: najbolje pre pocetka razvoja bolesti')
print('\t\tKarenca: za vinovu lozu 42 dana, a za krompir 14 dana')
print('\t- Preparat Everest (Chemical Agrosava):')
print('\t\tKolicina primene: 0,3-0,4% koncentracije za vinovu lozu i krompir')
print('\t\tVreme primene: za krompir od faze 2-3 lista do faze 3-5 cvetova, a za vinovu lozu u fazi intenzivnog porasta, cvetanja i razvoja bobica')
print('\t\tKarenca: za vinovu lozu 28 dana, a za krompir 7 dana')
print('\t- Preparat Cuproxat (Delta Agrar):')
print('\t\tKolicina primene: 0,2-0,35% koncentracije za vinovu lozu, odnosno 0,2% za krompir')
print('\t\tVreme primene: za vinovu lozu pri pojavi prvih znakova, a pre ostvarivanja uslova za sekundarne infekcije, a za krompir pre ostvarivanja uslova za zarazu')
print('\t\tKarenca: za stone sorte vinove loze 21 dan, za vinske sorte vinove loze 28 dana, a za krompir 7 dana')
print('\t- Preparat Antracol (Bayer):')
print('\t\tKolicina primene: 0,2% koncentracije za vinovu lozu, odnosno 1,5-1,8 kg/ha za krompir')
print('\t\tVreme primene: u oba slucaja pre pocetka infekcije, preventivno')
print('\t\tKarenca: za stone sorte vinove loze 21 dan, za vinske sorte vinove loze 42 dana, a za krompir 14 dana')

for i in range(0, brojKr + brojVl):
    if i <= brojVl - 1:
        img = imread('slike/vl' + i.__str__() + '.jpg')
        vrsta = 0
    else:
        img = imread('slike/kr' + i.__str__() + '.jpg')
        vrsta = 1
    hsl, wsl = img.shape[:2]
    imgGray = rgb2gray(img)
    thresh = threshold_otsu(imgGray)
    imgTreshold = imgGray <= thresh
# imgTreshold = 1 - threshold_adaptive(imgGray, block_size=75, offset=0.04)
#    plt.imshow(imgTreshold, 'gray')
    str_elem = disk(9)
    imgClosed = closing(imgTreshold, str_elem)
    imgClosed2 = closing(imgClosed, str_elem)
    labeledImg = label(imgClosed2)
    regions = regionprops(labeledImg)
    # print('Regioni:{}'.format(len(regions)))
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
                # print('biljka je vinova loza')
#             plt.imshow(draw_regions(regions_vineleaf, imgClosed2.shape), 'gray')
        else:
            if h > hsl / 2:
                regions_krompir.append(region)
                newline = str(ratio) + ',' + str(region.eccentricity) + ',' + str(region.solidity) + ',' + str(vrsta)
                txt_to_write.append('\n' + newline)
                # print('biljka je krompir')
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
    print 'Trening set: ' + repr(len(trainingSet))  # broj training pdoataka
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
        print('> dobijena vrednost')
    #    for y in range(len(result)):
        print '>>'+repr(result)
        print('>>>>> stvarna vrednost=' + repr(testSet[x][-1]))
    accuracy = get_accuracy(testSet, predictionsForK)
    print('Procenat tacnosti prepoznavanja vrste biljke je: ' + repr(accuracy) + '%')

    # img = cv2.imread('slike/kr20.jpg')
    # Z = img.reshape((-1, 3))

    # convert to np.float32
    # Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # K = 3
    # ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    # center = np.uint8(center)
    # res = center[label.flatten()]
    # res2 = res.reshape((img.shape))

    # cv2.imshow('res2', res2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


main()
