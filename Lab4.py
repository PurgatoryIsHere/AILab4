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
    root = None
    att_arr = []
    predictions = []
    tokens = []
    token_index = 0

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

def next_token():
        nonlocal token_index
        if token_index < len(tokens):
            token = tokens[token_index]
            token_index += 1
            return token
        return None

def read_node():
        # read att for node
        n = next_token()
        if n is None:
            return None

        if n[0] == '[':  # build return node
            return TreeNode(parent=None, attribute=None, children=None, return_val=n[1:-1])

        # build interior node
        node = TreeNode(parent=None, attribute=n, children={}, return_val=None)

        next_token_val = next_token()  # read (
        if next_token_val != "(":
            # Handle malformed model file
            return node

        val = next_token()
        while val != ")" and val is not None:
            child_node = read_node()
            if child_node is not None:
                node.children[val] = child_node
            val = next_token()
            if val is None:
                break

        return node

def read_model(modelfile):
        nonlocal root, att_arr, tokens, token_index
        try:
            with open(modelfile, 'r') as infile:
                # Read the first line containing attributes
                first_line = infile.readline().strip()
                if first_line:
                    att_arr = first_line.split()
                else:
                    att_arr = []

    content = " ".join(line.strip() for line in infile)
    tokens = content.split()
    token_index = 0

    root = read_node()

    # If root is None, create a default return node
    if root is None:
        root = TreeNode(parent=None, attribute=None, children=None, return_val="undefined")

    except Exception as e:
    print(f"Error reading model: {e}")
    # Create a default root node instead of exiting
    root = TreeNode(parent=None, attribute=None, children=None, return_val="undefined")

    def trace_tree(node, data):
        if node.return_val is not None:
            return node.return_val

        att = node.attribute
        try:
            att_index = att_arr.index(att)
            if att_index < len(data):
                val = data[att_index]
                if val in node.children:
                    t = node.children.get(val)
                    return trace_tree(t, data)
                else:
                    # Handle case where the value is not found in children
                    # Return the most common class in this node's children
                    class_counts = {}
                    for child in node.children.values():
                        if child.return_val is not None:
                            class_counts[child.return_val] = class_counts.get(child.return_val, 0) + 1

                    if class_counts:
                        return max(class_counts.items(), key=lambda x: x[1])[0]
                    else:
                        # If no clear majority, return first child's result
                        first_child = next(iter(node.children.values()))
                        return trace_tree(first_child, data)
            else:
                # Handle case where attribute index is out of bounds
                # Return most common return_val among children
                return next(iter(node.children.values())).return_val if node.children else "undefined"
        except (ValueError, IndexError):
            # Handle cases where attribute is not found or index error
            return "undefined"

    def predict_from_model(testfile):
        nonlocal predictions
        try:
            predictions = []
            with open(testfile, 'r') as file:
                for line in file:
                    tokens = line.strip().split()

                    if not tokens:  # Skip empty lines
                        continue

                    data = []
                    if tokens[0] == "-1":  # Check if first token is -1
                        tokens.pop(0)  # consume -1

                    # Match the length of data to att_arr
                    for i in range(len(att_arr)):
                        if i < len(tokens):
                            data.append(tokens[i])
                        else:
                            # Pad with empty string if not enough tokens
                            data.append("")

                    pred = trace_tree(root, data)
                    predictions.append(pred)

        except Exception as e:
            print(f"Error reading test file: {e}")
            # Don't exit, just return empty predictions
            predictions = []

    def save_predictions(outputfile):
        try:
            with open(outputfile, 'w') as p:
                for pred in predictions:
                    p.write(f"{pred}\n")
        except Exception as e:
            print(f"Error writing to file: {e}")

    # Execute the prediction process
    read_model(model)
    predict_from_model(data)
    save_predictions(prediction)



                
        
    


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