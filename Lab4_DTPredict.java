/*
This is the reference Java version of decision tree!
You can only use it for CS 330
- Dr. Cao
*/


import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;


public class DTPredict {
    private TreeNode root;
    private ArrayList<String> attArr;
    private ArrayList<String> predictions;

    public void readModel(String modelfile) {
        try {
            Scanner infile = new Scanner(new File(modelfile));
            String[] atts = infile.nextLine().split(" ");
            attArr = new ArrayList<String>();
            for(int i = 0; i < atts.length; i++)
                attArr.add(atts[i]);
            root = readNode(infile);
        }catch(IOException e) {
            System.out.println("Error reading model: " + e);
            System.exit(1);
        }
    }

    private TreeNode readNode(Scanner infile) {
        //read att for node
        String n = infile.next();
        if(n.charAt(0) == '[') { //build return node
            return new TreeNode(null, null, n.substring(1, n.length()-1));
        }
        //build interior node
        TreeNode node = new TreeNode(n, new HashMap<>(), null);
        infile.next();//read (
        String val = infile.next();
        while(!val.equals(")")) {
           // System.out.println("VAL: " + val);
            node.children.put(val, readNode(infile));
            val = infile.next();
        }
        return node;
    }

    public void predictFromModel(String testfile) {
        try {
            predictions = new ArrayList<>();
            Scanner s = new Scanner(new File(testfile));
            ArrayList<String> data;
            while(s.hasNext()) {
                data = new ArrayList<>();
                s.next(); //consume -1
                for(int i = 0; i<attArr.size(); i++)
                    data.add(s.next());

                String pred = traceTree(root, data);
                predictions.add(pred);
            }
        }catch(IOException e) {
            System.out.println("Error reading test file: " + e);
            System.exit(1);
        }
    }

    private String traceTree(TreeNode node, ArrayList<String> data) {
        if(node.returnVal !=null)
            return node.returnVal;
        String att = node.attribute;
        String val = data.get(attArr.indexOf(att));
        TreeNode t = node.children.get(val);
        return traceTree(t, data);
    }

    public void savePredictions(String outputfile) {
        try {
            PrintWriter p = new PrintWriter(outputfile);
            for(int i = 0; i < predictions.size(); i++)
                p.println(predictions.get(i));
            p.close();
        } catch(IOException e) {
            System.out.println("Error writing to file: " + e);
        }
    }

    public static void main(String[] args) {
        if(args.length == 3) {
            DTPredict t = new DTPredict();
            t.readModel(args[1]);
            System.out.println("Model read successfully");
            t.predictFromModel(args[0]);
            System.out.println("Predictions complete");
            t.savePredictions(args[2]);
            System.out.print("Predictions saved to file " + args[2]);
        }
        else
            System.out.println("Please format your input: DTTrain 'testdata' 'modelfile' 'outputfile'");
    }

    private class TreeNode {
        String attribute;
        HashMap<String, TreeNode> children;
        String returnVal;

        public TreeNode(String a, HashMap<String, TreeNode> h, String r) {
            attribute = a;
            children = h;
            returnVal = r;
        }
    }
}