import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;

public class NeuralNetwork {

    private List<NeuronLayer> layers = new LinkedList<>();
    private INDArray trainingSetOutputs;
    private INDArray trainingSetInputs;

    List<NeuronLayer> getLayers() {
        return layers;
    }

    NeuralNetwork(LinkedList<NeuronLayer> layers) {
        this.layers = layers;
        for (int x = 1; x <= layers.size(); x++) {
            final NeuronLayer layer = layers.get(x - 1);
            if (x == 1) {
                layer.setLayerName("Input layer");
            } else if (x == layers.size()) {
                layer.setLayerName("Output layer");
                layer.setIsOutputLayer(true);
            } else {
                layer.setLayerName("Layer " + x);
            }
        }
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT);
    }

    void validateNetworkSetup(INDArray trainingSetInputs, INDArray trainingSetOutputs) throws InvalidLayerSetupException {

        System.out.println("Training set inputs: " + trainingSetInputs.size(1));

        //validate layer setup
        for (int x = 0; x < layers.size(); x++) {
            NeuronLayer layer = layers.get(x);
            System.out.println("Layer: " + layer.getLayerName() + " NumInputs: " + layer.getNumberOfInputs() + " NumHiddenNodes: " + layer.getNumberOfHiddenNodes());
            int numInputs = (x == 0) ? trainingSetInputs.size(1) : layers.get(x - 1).getNumberOfHiddenNodes();
            if (numInputs != layer.getNumberOfInputs()) {
                throw new InvalidLayerSetupException("Layer " + (x + 1) + " has an invalid number of inputs.  Specified: " + numInputs + " Needs: " + layer.getNumberOfInputs());
            }
        }

        //ensure that output layer has correct number of neurons
    }

    // We train the neural network through a process of trial and error.
    // Adjusting the synaptic weights each time.
    void train(INDArray trainingSetInputs, INDArray trainingSetOutputs, int numberOfIterations) throws InvalidLayerSetupException {

        this.trainingSetInputs = trainingSetInputs;
        this.trainingSetOutputs = trainingSetOutputs;

        validateNetworkSetup(trainingSetInputs, trainingSetOutputs);
        for (int iteration = 0; iteration < numberOfIterations; iteration++) {

            // Pass the training set through our neural network
            passDataThroughNetwork(trainingSetInputs);

            //iterate through the network backwards and calculate the errors, deltas and adjustments
            ListIterator<NeuronLayer> listIterator = layers.listIterator(layers.size());

            INDArray error;
            INDArray outputs = trainingSetOutputs;
            NeuronLayer previousLayer = null;
            //iterate through the layers backwards and calculate error and delta for each layer
            while (listIterator.hasPrevious()) {
                NeuronLayer currentLayer = listIterator.previous();
//                System.out.println("Calculating error for layer: " + currentLayer.getLayerName());
                if (currentLayer.isOutputLayer()) {
//                    System.out.println("Subtracting iteration outputs from training set outputs");
//                    System.out.println("Training set outputs size: " + outputs.size(0) + " by " + outputs.size(1));
//                    System.out.println("Network outputs size: " + currentLayer.getIterationOutputs().size(0) + " by " + currentLayer.getIterationOutputs().size(1));

                    error = outputs.sub(currentLayer.getIterationOutputs());
                    System.out.println("Iteration: " + iteration + " Avg. Error: " + String.format("%.12f", MathHelpers.meanOfVector(error)));
                    currentLayer.setIterationDelta(error.mul(MathHelpers.sigmoidDerivative(currentLayer.getIterationOutputs(), true)));
                } else {
                    error = previousLayer.getIterationDelta().mmul(previousLayer.getSynapticWeights().transpose());
                    currentLayer.setIterationDelta(error.mul(MathHelpers.sigmoidDerivative(currentLayer.getIterationOutputs(), true)));
                }
                previousLayer = currentLayer;
            }

            //now we iterate forward and calculate the adjustment for each layer
            INDArray inputs = trainingSetInputs;
            for (NeuronLayer layer : layers) {
                INDArray adjustment = inputs.transpose().mmul(layer.getIterationDelta());
                inputs = layer.getIterationOutputs();
                layer.setSynapticWeights(layer.getSynapticWeights().add(adjustment));
            }

        }
    }

    // The neural network thinks.
    void passDataThroughNetwork(INDArray inputs) {
        for (NeuronLayer layer : layers) {
            //apply weights and activation function to each layer
            //the output of one layer serves as the input to the next
            INDArray weightedResult = layer.applyWeights(inputs);
            inputs = layer.applyActivationFunction(weightedResult);
        }
    }

    // The neural network System.out.printlns its weights
    void print_weights() {
        for (NeuronLayer layer : layers) {
//            System.out.println(layer.getLayerName());
//            System.out.println(ND4JHelpers.getPrintableValues(layer.getSynapticWeights()));
        }
    }

    class InvalidLayerSetupException extends Exception {
        public InvalidLayerSetupException(String message) {
            super(message);
        }
    }

    public INDArray getResults() {
        NeuronLayer outputLayer = layers.get(layers.size() - 1);
        return outputLayer.getIterationOutputs();
    }

}