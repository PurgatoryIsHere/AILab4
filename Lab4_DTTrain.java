/*
This is the reference Java version of decision tree!
You can only use it for CS 330
*/

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;


public class DTTrain {
    private HashMap<String,ArrayList<ArrayList<String>>> datamap; //stores all data read in
    private HashMap<String, ArrayList<String>> attvalues;
    private String[] atts; //A list of attributes read in. atts[0] is the classifier
    private int numAtts; //The number of attributes used to predict atts[0]

    private int numClasses; //The total number of classes to predict between

    private TreeNode root;

    public void readFile(String infile, int percent) {
        try {
            //initialize map for storing data
            datamap = new HashMap<String,ArrayList<ArrayList<String>>>();
            //open the training data file
            Scanner readFile = new Scanner(new File(infile));
            //Read the attributes from the first line of the file
            String attline = readFile.nextLine().substring(1);
            atts = attline.split("\\|");
            numAtts = atts.length-1;
            //initialize map of attribute values
            attvalues = new HashMap<String, ArrayList<String>>();
            for(String a: atts)
                attvalues.put(a, new ArrayList<String>());

            //read data into map
            int index = 0;
            while(readFile.hasNext()) {
                String dataclass = readFile.next(); //read in classification for data

                ArrayList<String> arr = attvalues.get(atts[0]);
                if(!arr.contains(dataclass))
                    arr.add(dataclass);

                if(!datamap.containsKey(dataclass))
                    datamap.put(dataclass, new ArrayList<ArrayList<String>>());
                ArrayList<ArrayList<String>> a = datamap.get(dataclass);
                ArrayList<String> datapoint = new ArrayList<String>();
                for(int i = 0; i < numAtts; i++) { //for each attribute
                    String val = readFile.next();
                    datapoint.add(val); //put data point into datamap
                    //add val to list of possible outcomes for attribute
                    arr = attvalues.get(atts[i+1]);
                    if(!arr.contains(val))
                        arr.add(val);
                }
                //only add data point to map percent of the time
                if(index%100 < percent)
                    a.add(datapoint);

                index++;
            }//while

            numClasses = datamap.keySet().size();
        }
        catch(IOException e) {
            System.out.println("Error reading file : " + e);
            System.exit(0);
        }
    }


    public void buildTree() {
        root = new TreeNode(null);

        ArrayList<String> currFreeAtts = new ArrayList<String>();
        for(int i = 0; i < numAtts; i++)
            currFreeAtts.add(atts[i+1]);

        root = buildTreeNode(null, currFreeAtts, datamap);
    }


    private TreeNode buildTreeNode(TreeNode parent, ArrayList<String> currFreeAtts,
                                   HashMap<String,ArrayList<ArrayList<String>>> nodeData) {
        //build current node
        TreeNode curr = new TreeNode(parent);

        double minEnt = 1;
        String minAtt = null;
        //calc current entropy for each attribute
        for(int i = 0; i < numAtts; i++) { //for each attribute
            String att = currFreeAtts.get(i);//get the attribute
            if (att != null) {//if the attribute hasn't already been used in the tree
                ArrayList<String> vals = attvalues.get(att); //get the list of possible values of the attribute

                int[][] partition = new int[vals.size()][numClasses]; //store class counts for each outcome
                for(int j = 0; j < numClasses; j++) { //for each classification
                    String outcome = attvalues.get(atts[0]).get(j);
                    ArrayList<ArrayList<String>> l = nodeData.get(outcome);
                    for(ArrayList<String> l2: l) {
                        partition[vals.indexOf(l2.get(i))][j] += 1;
                    }//for
                }//for
                //calculate entropy
                double ent = partitionEntropy(partition);
                //System.out.println(att + ent);
                if(ent < minEnt) {
                    minEnt = ent;
                    minAtt = att;
                }
            }//if
        }//for

        //if at base of tree
        if(minAtt == null) {
            int max = 0;
            String maxClass = "undefined";
            for(int j = 0; j < numClasses; j++) { //for each classification
                String outcome = attvalues.get(atts[0]).get(j);
                if (nodeData.get(outcome).size() >= max) {
                    max = nodeData.get(outcome).size();
                    maxClass = outcome;
                }
            }
            //System.out.println(maxClass);
            curr.returnVal = maxClass;
            return curr;
        }
        //System.out.println(minAtt);
        //find best attribute
        curr.attribute = minAtt;
        int attindex = currFreeAtts.indexOf(minAtt);
        currFreeAtts.set(attindex, null);

        //build child nodes
        for(String v: attvalues.get(minAtt)) {
            HashMap<String,ArrayList<ArrayList<String>>> tempmap = new HashMap<>();
            for(int j = 0; j < numClasses; j++) { //for each classification
                String outcome = attvalues.get(atts[0]).get(j);
                ArrayList<ArrayList<String>> trimList = new ArrayList<>();
                ArrayList<ArrayList<String>> l = nodeData.get(outcome);
                for(ArrayList<String> l2: l) {
                    if(l2.get(attindex).equals(v))
                        trimList.add(l2);
                }
                tempmap.put(outcome, trimList);

            }//for
            System.out.print(v + "---> ");
            curr.children.put(v,buildTreeNode(curr,currFreeAtts,tempmap));
        }
        //return built node
        currFreeAtts.set(attindex, minAtt);
        return curr; //return built node
    }

    public double partitionEntropy(int[][] partition) {
        double totalEnt = 0;
        double total = 0;

        for(int i = 0; i < partition.length; i++) {
            double n = 0;
            for (int j = 0; j < partition[0].length; j++) {
                n += partition[i][j];
                total += partition[i][j];
            }
            totalEnt += n*entropy(partition[i]);

        }

        return totalEnt/total;
    }

    public double entropy(int[] classCounts) {
        double total = 0;
        for(int i: classCounts)
            total += i;

        double sum = 0;
        for(int i = 0; i < classCounts.length; i++)
            sum -= (classCounts[i]/total)*log2(classCounts[i]/total);

        return sum;
    }

    public double log2(double x) {
        if(x == 0)
            return 0;
        return Math.log(x)/Math.log(2);
    }

    public void saveModel(String modelfile) {
        try {
            PrintWriter outfile = new PrintWriter(modelfile);

            for(int i = 0; i < numAtts; i++)
                outfile.print(atts[i+1] + " ");
            outfile.println();

            writeNode(outfile, root);

            outfile.close();

        }catch(IOException e) {
            System.out.println("Error writing to file: " + e);
            System.exit(1);
        }
    }

    public void writeNode(PrintWriter outfile, TreeNode curr) {
        if(curr.returnVal != null) {
            outfile.print("[" + curr.returnVal + "] ");
            return;
        }
        outfile.print(curr.attribute + " ( ");
        for(Map.Entry<String, TreeNode> en: curr.children.entrySet()) {
            outfile.print(en.getKey() + " ");
            writeNode(outfile, en.getValue());
        }

        outfile.print(" ) ");
    }

    public static void main(String[] args) {
        if (args.length >= 2) {
            String infile = args[0];
            String modelfile = args[1];
            DTTrain t = new DTTrain();
            if(args.length == 3)
                t.readFile(infile, Integer.parseInt(args[2]));
            else
                t.readFile(infile,100);

            t.buildTree();

            t.saveModel(modelfile);
        }
        else {
            System.out.println("Please format your input: DTTrain 'trainingdata' 'modelfile' [percentage of data to train on]");
        }
    }

    private class TreeNode{
        TreeNode parent;
        HashMap<String, TreeNode> children;
        String attribute;
        String returnVal;

        public TreeNode(TreeNode p) {
            parent = p;
            attribute = "none";
            children = new HashMap<String, TreeNode>();
            returnVal = null;
        }
    }
}