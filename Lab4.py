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
    
    def build_tree_node(parent, curr_free_atts, node_data, attvalues, features, classes):
        curr_node = TreeNode(parent)
        curr_node.class_counts = defaultdict(int)
        total = 0
        for cls in classes:
            count = len(node_data.get(cls, []))
            curr_node.class_counts[cls] = count
            total += count

        if total == 0:
            if parent:
                curr_node.returnVal = majority_class(parent.class_counts, classes)
            else:
                curr_node.returnVal = classes[0] if classes else 'unknown'
            return curr_node

        if len([cnt for cnt in curr_node.class_counts.values() if cnt > 0]) == 1:
            curr_node.returnVal = max(curr_node.class_counts, key=lambda k: curr_node.class_counts[k])
            return curr_node

        available_attrs = [a for a in curr_free_atts if a is not None]
        if not available_attrs:
            curr_node.returnVal = majority_class(curr_node.class_counts, classes)
            return curr_node

        min_entropy = float('inf')
        best_attr = None
        for attr in available_attrs:
            attr_idx = features.index(attr)
            values = attvalues[attr]
            partition = []
            for v in values:
                part_counts = [0] * len(classes)
                for cls_idx, cls in enumerate(classes):
                    for dp in node_data.get(cls, []):
                        if dp[attr_idx] == v:
                            part_counts[cls_idx] += 1
                partition.append(part_counts)
            ent = partition_entropy(partition)
            if ent < min_entropy:
                min_entropy = ent
                best_attr = attr

        if best_attr is None:
            curr_node.returnVal = majority_class(curr_node.class_counts, classes)
            return curr_node

        curr_node.attribute = best_attr
        attindex = curr_free_atts.index(best_attr)
        curr_free_atts[attindex] = None

        best_attr_idx = features.index(best_attr)
        for v in attvalues[best_attr]:
            child_data = defaultdict(list)
            for cls in classes:
                for dp in node_data.get(cls, []):
                    if dp[best_attr_idx] == v:
                        child_data[cls].append(dp)
            child_node = build_tree_node(curr_node, curr_free_atts.copy(), child_data, attvalues, features, classes)
            curr_node.children[v] = child_node

        curr_free_atts[attindex] = best_attr
        return curr_node

    def build_tree():
        nonlocal root
        curr_free_atts = []
        for i in range(numAtts):
            curr_free_atts.append(atts[i + 1])

        # Create the initial data structure for all classes
        node_data = {}
        for cls in datamap:
            node_data[cls] = datamap[cls]

        # Call build_tree_node with the proper parameters
        root = build_tree_node(
            None,
            curr_free_atts,
            node_data,
            attvalues,
            atts[1:],
            list(datamap.keys())
        )

    def majority_class(class_counts, classes):
        max_count = -1
        max_class = None
        for cls in classes:
            if class_counts.get(cls, 0) > max_count:
                max_count = class_counts.get(cls, 0)
                max_class = cls
        return max_class if max_class else classes[0] if classes else "unknown"

    def write_node(outfile, curr):
        if curr.return_val is not None:
            outfile.write(f"[{curr.return_val}] ")
            return

        outfile.write(f"{curr.attribute} ( ")
        for key, value in curr.children.items():
            outfile.write(f"{key} ")
            write_node(outfile, value)

        outfile.write(" ) ")

    def save_model(modelfile):
        try:
            with open(modelfile, 'w') as outfile:
                for i in range(numAtts):
                    outfile.write(f"{atts[i + 1]} ")
                outfile.write("\n")

                write_node(outfile, root)

        except Exception as e:
            print(f"Error writing to file: {e}")
            exit(1)

    # Execute the training process
    read_file(data, 100)  # Using 100% of data by default
    build_tree()
    save_model(model)

