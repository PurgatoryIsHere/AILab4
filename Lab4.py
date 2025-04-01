"""
 Name: Evan Archer
 Assignment: Lab 4 - Decision Tree
 Course: CS 330
 Semester: Spring 2025
 Instructor: Dr. Cao
 Date: March 20th, 2025
 Sources consulted:

 Known Bugs:

 Creativity:

 Instructions: After a lot of practice in Python, in this lab, you are going to design the program for decision tree and implement it from scrath! Don't be panic, you still have some reference, actually you are going to translate the JAVA code to Python! The format should be similar to Lab 2!

"""
import sys
import argparse
import math
import os
from collections import defaultdict

class TreeNode:
    def __init__(self, parent=None, attribute=None, children=None, return_val=None):
        self.parent = parent
        self.attribute = attribute
        self.children = children if children is not None else {}
        self.return_val = return_val


def DTtrain(data, model):
    """
    This is the function for training a decision tree model
    """
    pass


def DTpredict(data, model, prediction):
    """
    This is the main function to make predictions on the test dataset. It will load saved model file,
    and also load testing data TestDataNoLabel.txt, and apply the trained model to make predictions.
    You should save your predictions in prediction file, each line would be a label, such as:
    1
    0
    0
    1
    ...
    """
    pass


def EvaDT(predictionLabel, realLabel, output):
    """
    This is the main function. You should compare line by line,
     and calculate how many predictions are correct, how many predictions are not correct. The output could be:

    In total, there are ??? predictions. ??? are correct, and ??? are not correct.

    """
    correct, incorrect, length = 0, 0, 0
    with open(predictionLabel, 'r') as file1, open(realLabel, 'r') as file2:
        pred = [line for line in file1]
        real = [line for line in file2]
        length = len(pred)
        for i in range(length):
            if pred.pop(0) == real.pop(0):
                correct += 1
            else:
                incorrect += 1
    Rate = correct / length

    result = "In total, there are " + str(length) + " predictions. " + str(correct) + " are correct and " + str(
        incorrect) + " are incorrect. The percentage is " + str(Rate)
    with open(output, "w") as fh:
        fh.write(result)


def main():
    options = parser.parse_args()
    mode = options.mode  # first get the mode
    print("mode is " + mode)
    if mode == "T":
        """
        The training mode
        """
        inputFile = options.input
        outModel = options.output
        if inputFile == '' or outModel == '':
            showHelper()
        DTtrain(inputFile, outModel)
    elif mode == "P":
        """
        The prediction mode
        """
        inputFile = options.input
        modelPath = options.modelPath
        outPrediction = options.output
        if inputFile == '' or modelPath == '' or outPrediction == '':
            showHelper()
        DTpredict(inputFile, modelPath, outPrediction)
    elif mode == "E":
        """
        The evaluating mode
        """
        predictionLabel = options.input
        trueLabel = options.trueLabel
        outPerf = options.output
        if predictionLabel == '' or trueLabel == '' or outPerf == '':
            showHelper()
        EvaDT(predictionLabel, trueLabel, outPerf)
    pass


def showHelper():
    parser.print_help(sys.stderr)
    print("Please provide input augument. Here are examples:")
    print("python " + sys.argv[0] + " --mode T --input TrainingData.txt --output DTModel.txt")
    print("python " + sys.argv[
        0] + " --mode P --input TestDataNoLabel.txt --modelPath DTModel.txt --output TestDataLabelPrediction.txt")
    print("python " + sys.argv[
        0] + " --mode E --input TestDataLabelPrediction.txt --trueLabel LabelForTest.txt --output Performance.txt")
    sys.exit(0)


if __name__ == "__main__":
    # ------------------------arguments------------------------------#
    # Shows help to the users                                        #
    # ---------------------------------------------------------------#
    parser = argparse.ArgumentParser()
    parser._optionals.title = "Arguments"
    parser.add_argument('--mode', dest='mode',
                        default='',  # default empty!
                        help='Mode: T for training, and P for making predictions, and E for evaluating the machine learning model')
    parser.add_argument('--input', dest='input',
                        default='',  # default empty!
                        help='The input file. For T mode, this is the training data, for P mode, this is the test data without label, for E mode, this is the predicted labels')
    parser.add_argument('--output', dest='output',
                        default='',  # default empty!
                        help='The output file. For T mode, this is the model path, for P mode, this is the prediction result, for E mode, this is the final result of evaluation')
    parser.add_argument('--modelPath', dest='modelPath',
                        default='',  # default empty!
                        help='The path of the machine learning model ')
    parser.add_argument('--trueLabel', dest='trueLabel',
                        default='',  # default empty!
                        help='The path of the correct label ')
    if len(sys.argv) < 3:
        showHelper()
    main()