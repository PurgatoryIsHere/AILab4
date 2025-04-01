"""
 Name: Evan Archer, Gabe Sanders, Christopher Bryce, Marvin Wocheslander
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
    # Initialize the DTTrain class-like structure
    datamap = {}  # stores all data read in
    attvalues = {}
    atts = []  # A list of attributes read in. atts[0] is the classifier
    numAtts = 0  # The number of attributes used to predict atts[0]
    numClasses = 0  # The total number of classes to predict between
    root = None

    def read_file(infile, percent):
        nonlocal datamap, attvalues, atts, numAtts, numClasses
        try:
            # initialize map for storing data
            datamap = {}

            # open the training data file
            with open(infile, 'r') as file:
                # Read the attributes from the first line of the file
                attline = file.readline().strip()[1:]
                atts = attline.split('|')
                numAtts = len(atts) - 1

                # initialize map of attribute values
                attvalues = {}
                for a in atts:
                    attvalues[a] = []

                # read data into map
                index = 0
                for line in file:
                    data_values = line.strip().split()
                    dataclass = data_values[0]  # read in classification for data

                    arr = attvalues[atts[0]]
                    if dataclass not in arr:
                        arr.append(dataclass)

                    if dataclass not in datamap:
                        datamap[dataclass] = []

                    a = datamap[dataclass]
                    datapoint = []

                    for i in range(numAtts):  # for each attribute
                        val = data_values[i + 1]
                        datapoint.append(val)  # put data point into datamap

                        # add val to list of possible outcomes for attribute
                        arr = attvalues[atts[i + 1]]
                        if val not in arr:
                            arr.append(val)

                    # only add data point to map percent of the time
                    if index % 100 < percent:
                        a.append(datapoint)

                    index += 1

            numClasses = len(datamap.keys())

        except Exception as e:
            print(f"Error reading file: {e}")
            exit(0)

    def log2(x):
        if x == 0:
            return 0
        return math.log(x) / math.log(2)

    def entropy(class_counts):
        total = 0
        for i in class_counts:
            total += i

        sum_val = 0
        for i in range(len(class_counts)):
            if total == 0:
                continue
            prob = class_counts[i] / total
            sum_val -= prob * log2(prob)

        return sum_val

    def partition_entropy(partition):
        total_ent = 0
        total = 0

        for i in range(len(partition)):
            n = 0
            for j in range(len(partition[0])):
                n += partition[i][j]
                total += partition[i][j]

        if total == 0:
            return 0

        for i in range(len(partition)):
            n = 0
            for j in range(len(partition[0])):
                n += partition[i][j]

            if n > 0:  # Avoid division by zero
                total_ent += n * entropy(partition[i])

        return total_ent / total
