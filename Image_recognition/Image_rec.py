import numpy as np
from PIL import Image # If 32 bit OS use import Image
import matplotlib.pyplot as plt
import time
import functools
from collections import Counter


def createExamples(): # For making the database of all sample images arrays
    numberArrayExamples = open('numArEx.txt','w')
    numbersWeHave = range(0,10) # 0-9
    versionsWeHave = range(1,10) # 1-9

    for eachNum in numbersWeHave:
        for eachVersion in versionsWeHave:
            imageFilePath = 'tutorialimages/images/numbers/'+str(eachNum)+'.'+str(eachVersion)+'.png'
            ei = Image.open(imageFilePath) # Open exampleImage
            eiAr = np.array(ei) # Example Image Array
            eiArl = str(eiAr.tolist()) # COnvert array to a list

            lineToWrite = str(eachNum)+'::'+eiAr1+'\n'
            numberArrayExamples.write(lineToWrite)

def threshold(imageArray): # Convert every image to a black and white image
    balanceAr = []
    newAr = imageArray
    newAr.flags.writeable = True

    for eachRow in imageArray:
        for eachPixel in eachRow:
            avgNum = functools.reduce(lambda x, y: x+y, eachPixel[:3])/len(eachPixel[:3])
            balanceAr.append(avgNum)
    balance = functools.reduce(lambda x, y: x+y, balanceAr)/len(balanceAr)

    for eachRow in newAr:
        for eachPix in eachRow:
            if functools.reduce(lambda x, y: x+y, eachPix[:3])/len(eachPix[:3])>balance:
                eachPix[0] = 255
                eachPix[1] = 255
                eachPix[2] = 255
                eachPix[3] = 255

            else:
                eachPix[0] = 0
                eachPix[1] = 0
                eachPix[2] = 0
                eachPix[3] = 255

    return newAr


def whatNumIsThis(filePath):
    matchedAr = []
    loadExamples = open('numArEx.txt','r').read()
    loadExamples = loadExamples.split('\n')

    i = Image.open(filePath)
    iAr = np.array(i)
    iArl = iAr.tolist()

    inQuestion = str(iArl)


    for eachExample in loadExamples:
        if len(eachExample) > 3:
            splitEx = eachExample.split('::')
            currentNum = splitEx[0]
            currentAr = splitEx[1]

            eachPixEx = currentAr.split('],')

            eachPixInQ = inQuestion.split('],')

            x = 0

            while x < len(eachPixEx):
                if eachPixEx[x] == eachPixInQ[x]:
                    matchedAr.append(int(currentNum))

                x+=1
         

    print(matchedAr)
    x = Counter(matchedAr)
    print(x)

    graphX = []
    graphY = []

    for eachThing in x:
        print(eachThing)
        graphX.append(eachThing)
        print(x[eachThing])
        graphY.append(x[eachThing])

    fig = plt.figure()
    ax1 = plt.subplot2grid((4,4), (0,0), rowspan=1, colspan=4)
    ax2 = plt.subplot2grid((4,4), (1,0), rowspan=3, colspan=4)

    ax1.imshow(iAr)
    ax2.bar(graphX,graphY, align='center')
    plt.ylim(400)

    xloc = plt.MaxNLocator(12)
    ax2.xaxis.set_major_locator(xloc)
    plt.show()

     

'''i = Image.open('tutorialimages/images/numbers/0.1.png')
imageArr = np.asarray(i)

i2 = Image.open('tutorialimages/images/numbers/y0.4.png')
imageArr2 = np.asarray(i2)

i3 = Image.open('tutorialimages/images/numbers/y0.5.png')
imageArr3 = np.asarray(i3)

i4 = Image.open('tutorialimages/images/sentdex.png')
imageArr4 = np.asarray(i4)

threshold(imageArr3)
threshold(imageArr2)
threshold(imageArr4)

fig = plt.figure()
ax1 = plt.subplot2grid((8,6), (0,0), rowspan=4, colspan=3)
ax2 = plt.subplot2grid((8,6), (4,0), rowspan=4, colspan=3)
ax3 = plt.subplot2grid((8,6), (0,3), rowspan=4, colspan=3)
ax4 = plt.subplot2grid((8,6), (4,3), rowspan=4, colspan=3)

ax1.imshow(imageArr)
ax2.imshow(imageArr2)
ax3.imshow(imageArr3)
ax4.imshow(imageArr4)

plt.show()'''


whatNumIsThis('tutorialimages/images/numbers/9.4.png')
