import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class NeuronLayer {

    private INDArray synapticWeights;
    private INDArray iterationOutputs;
    private INDArray iterationDelta;
    private INDArray iterationInputs;
    private String layerName;
    private int numberOfHiddenNodes;
    private int numberOfInputs;
    private boolean outputLayer;
//    private INDArray synapticWeights;

    public NeuronLayer(int numberOfHiddenNodes, int numberOfInputsPerNeuron) {
//        this.layerName = layerName;
        this.numberOfInputs = numberOfInputsPerNeuron;
        this.numberOfHiddenNodes = numberOfHiddenNodes;

        INDArray rand = Nd4j.rand(numberOfInputsPerNeuron, numberOfHiddenNodes);
        rand = rand.mul(2);
        INDArray ones = Nd4j.ones(numberOfInputsPerNeuron, numberOfHiddenNodes);
        rand = rand.sub(ones);

        this.synapticWeights =rand;// Nd4j.rand(numberOfInputsPerNeuron, numberOfHiddenNodes).mul(2).sub(Nd4j.ones(numberOfInputsPerNeuron, numberOfHiddenNodes));
    }

    public INDArray applyWeights(INDArray inputs) {
        iterationInputs = inputs;
//        iterationOutputs = MathHelpers.sigmoid(inputs.mmul(synapticWeights));
//        return iterationOutputs;
        return inputs.mmul(synapticWeights);
    }

    public INDArray applyActivationFunction(INDArray inputs) {
//        iterationInputs = inputs;
        iterationOutputs = MathHelpers.sigmoid(inputs);
        return iterationOutputs;
    }

    public INDArray getSynapticWeights() {
        return synapticWeights;
    }

    public void setSynapticWeights(INDArray synapticWeights) {
        this.synapticWeights = synapticWeights;
    }

    public INDArray getIterationOutputs() {
        return iterationOutputs;
    }

    public INDArray getIterationDelta() {
        return iterationDelta;
    }

    public void setIterationDelta(INDArray iterationDelta) {
        this.iterationDelta = iterationDelta;
    }

    public INDArray getIterationInputs() {
        return iterationInputs;
    }

    public void setIterationInputs(INDArray iterationInputs) {
        this.iterationInputs = iterationInputs;
    }

    public String getLayerName() {
        return layerName;
    }

    public int getNumberOfHiddenNodes() {
        return numberOfHiddenNodes;
    }

    public int getNumberOfInputs() {
        return numberOfInputs;
    }

    public void setLayerName(String layerName) {
        this.layerName = layerName;
    }

    public boolean isOutputLayer() {
        return outputLayer;
    }

    public void setIsOutputLayer(boolean outputLayer) {
        this.outputLayer = outputLayer;
    }
}
