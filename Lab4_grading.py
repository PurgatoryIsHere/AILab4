"""
 Grading code for Lab 4

 Written by Instructor: Dr. Cao
 Date: the day you read this script

 Instructions: This code is written by Dr. Cao for your reference. You may or may not use it, if you don't, your lab 4 grade could be a mystery :)

"""
import os
import sys
import math
import random
import os.path
from os import path
# first check if your Lab4.py is there

ProgramName = None
if path.exists("Lab4_solution.py") or path.exists("Lab4.py"):
    print("Great, the Lab4.py exists ")
    ProgramName = "Lab4_solution.py" if path.exists("Lab4_solution.py") else "Lab4.py"
else:
    print("Abort, please make sure Lab4.py is under the current directory")
    sys.exit(0)



def generateTraining(output,numFeature = None, FeatureValues = None, numRecord = None):
    """
    This function will randomly generate the training data
    """
    random.seed = 330
    if numFeature == None:
        numFeature = random.randrange(3,20)
    if FeatureValues == None:
        FeatureValues = random.randrange(0,100)
    if numRecord == None:
        numRecord = random.randrange(100,5000)
    # get the header
    fh = open(output,"w")
    fh.write("#output")
    for i in range(numFeature):
        fh.write("|attr"+str(i))
    fh.write("\n")
    # save each training record
    for r in range(numRecord):
        rOut = random.randrange(2)
        fh.write(str(rOut))
        for i in range(numFeature):
            rValue = random.randrange(FeatureValues)
            fh.write(" "+str(rValue))
        fh.write("\n")
    fh.close()

    return(numFeature,FeatureValues,numRecord)

def generateTest(numFeatureRange, featureValuesRange, output):
    """
    This function will generate random testing data without label
    """
    numRecord = random.randrange(100,5000)
    fh = open(output,"w")
    for r in range(numRecord):
        fh.write("-1")
        for i in range(numFeatureRange):
            rValue = random.randrange(featureValuesRange)
            fh.write(" "+str(rValue))
        fh.write("\n")
    fh.close()
    return numRecord
def deleteTemFile(tobeDeleted):
    if os.path.isfile(tobeDeleted):
        os.system("del "+tobeDeleted)

def verify(input, totalAmount):
    """
    Simply check if the total amount matches
    """
    with open(input,"r") as fh:
        for line in fh:
            if line.strip() == "":
                continue
            totalAmount-=1
    return totalAmount == 0


finalScore = 0

print("Preparing environment and start testing ...")
# normal and easy test
TrainingData = "CAOTRAINING_testTraining.txt"
TestingData = "CAOTEST_testNoLabel.txt"
ModelPath = "CAOMODEL.txt"
PredictionData = "CAOPREDICTION_testPred.txt"
deleteTemFile(TrainingData)
deleteTemFile(TestingData)
deleteTemFile(ModelPath)
deleteTemFile(PredictionData)



(N,F,NRecord) = generateTraining(TrainingData,numFeature=10,FeatureValues=10, numRecord=20)
NRecord = generateTest(N,F,TestingData)

print("Testing standard dataset ... ")
try:
    os.system("python "+ ProgramName + " --mode T --input "+TrainingData+" --output "+ModelPath)
except:
    print("error when running "+ProgramName)

try:
    os.system("python "+ ProgramName + " --mode P --input "+TestingData+" --modelPath "+ModelPath + " --output " + PredictionData)
    if verify(PredictionData,NRecord):
        print("Great! Your DT Training and prediction function seems working ...")
        finalScore+=3
    else:
        print("It seems your prediction didn't match with the total number of input? please check")
except:
    print("Error when running the prediction mode")


# now do more tests
deleteTemFile(TrainingData)
deleteTemFile(TestingData)
deleteTemFile(ModelPath)
deleteTemFile(PredictionData)


print("Now testing a little bit different training dataset ....")

(N,F,NRecord) = generateTraining(TrainingData,numFeature=20,FeatureValues=100, numRecord=20)
NRecord = generateTest(N,F,TestingData)

try:
    os.system("python "+ ProgramName + " --mode T --input "+TrainingData+" --output "+ModelPath)
except:
    print("error when running "+ProgramName)

try:
    os.system("python "+ ProgramName + " --mode P --input "+TestingData+" --modelPath "+ModelPath + " --output " + PredictionData)
    if verify(PredictionData,NRecord):
        print("Great! Your DT Training and prediction function seems working ...")
        finalScore+=3
    else:
        print("It seems your prediction didn't match with the total number of input? please check")
except:
    print("Error when running the prediction mode")


print("Now testing more complicated dataset ....")
deleteTemFile(TrainingData)
deleteTemFile(TestingData)
deleteTemFile(ModelPath)
deleteTemFile(PredictionData)
(N,F,NRecord) = generateTraining(TrainingData)
NRecord = generateTest(N,F,TestingData)

try:
    os.system("python "+ ProgramName + " --mode T --input "+TrainingData+" --output "+ModelPath)
except:
    print("error when running "+ProgramName)

try:
    os.system("python "+ ProgramName + " --mode P --input "+TestingData+" --modelPath "+ModelPath + " --output " + PredictionData)
    if verify(PredictionData,NRecord):
        print("Great! Your DT Training and prediction function seems working ...")
        finalScore+=4
    else:
        print("It seems your prediction didn't match with the total number of input? please check")
except:
    print("Error when running the prediction mode")


print("Now testing more complicated dataset ....")
deleteTemFile(TrainingData)
deleteTemFile(TestingData)
deleteTemFile(ModelPath)
deleteTemFile(PredictionData)
(N,F,NRecord) = generateTraining(TrainingData)
NRecord = generateTest(N,F,TestingData)

try:
    os.system("python "+ ProgramName + " --mode T --input "+TrainingData+" --output "+ModelPath)
except:
    print("error when running "+ProgramName)

try:
    os.system("python "+ ProgramName + " --mode P --input "+TestingData+" --modelPath "+ModelPath + " --output " + PredictionData)
    if verify(PredictionData,NRecord):
        print("Great! Your DT Training and prediction function seems working ...")
        finalScore+=4
    else:
        print("It seems your prediction didn't match with the total number of input? please check")
except:
    print("Error when running the prediction mode")


print("Your total score is " + str(finalScore))
print("The max you can get here is 14. You should get anther 2 points when you post your comment on discord, and another 4 points once you also show the performance of weka and your model on the provided training dataset")
print("Bonus points are possible (up to 3 points) if you add additional function to this lab, such as different entropy function in the decision tree")
deleteTemFile(TrainingData)
deleteTemFile(TestingData)
deleteTemFile(ModelPath)
deleteTemFile(PredictionData)