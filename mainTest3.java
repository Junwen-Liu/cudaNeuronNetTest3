
import java.io.Console;
import java.util.*;

class mainTest3 {
    public native boolean simpleNN(Map<String, String> testmap);
    // public native boolean simpleNN(int numOfLayers, int[][] layerSepcs);

    public static void main(String args[]) {
        mainTest3 test = new mainTest3();

        // construct the Neuron Network
        // In HashMap, Integer represent number of layer
        // in HashMap, int[0] specify:
        // 0: linear layer, int[1] indicates number of input neuron, int[2] indicates
        // number of output neuron.
        // 1: relu layer
        // 2: sigmoid layer

        Map<String, String> layerSpecs = new HashMap<String, String>();
        layerSpecs.put("0", "0,2,30"); // linear layer, input 2 neuron,output 30 neuron
        layerSpecs.put("1", "1"); // relu layer
        layerSpecs.put("2", "0,30,1"); // linear layer, input 30 neuron, output 2 nueron
        layerSpecs.put("3", "2"); // sigmoid layer

        // int numOfLayers = 4;
        // int[][] layersSpecs = { { 0, 2, 30 }, { 1 }, { 0, 30, 1 }, { 2 } };

        // boolean result = test.simpleNN(numOfLayers, layersSpecs);
        boolean result = test.simpleNN(layerSpecs);
        String checkresult = result ? "This cuda program has been running successfully"
                : "This cuda program has failed";
        System.out.println("Checking if GPU CUDA program has been running successfully:  \n" + checkresult);
    }
}
