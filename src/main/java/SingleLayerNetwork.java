import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class SingleLayerNetwork {

    private INDArray synaptic_weights;

    public SingleLayerNetwork() {

        // We model a single neuron, with 3 input connections and 1 output connection.
        // We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1 and mean 0.
        this.synaptic_weights = Nd4j.rand(3, 1).mul(2).sub(Nd4j.ones(3,1));
    }

    // We train the neural network through a process of trial and error adjusting the synaptic weights on each iteration.
    void train(INDArray trainingSetInputs, INDArray trainingSetOutputs, int numberOfTrainingIterations) {

        for (int iteration = 0; iteration < numberOfTrainingIterations; iteration++) {

            // Pass the training set through our neural network (a single neuron).
            INDArray output = think(trainingSetInputs);

            // Calculate the error (The difference between the desired output
            // and the predicted output).
            INDArray error = trainingSetOutputs.sub(output);
            System.out.println("Errors: " + error);

            // Multiply the error by the input and again by the gradient of the Sigmoid curve.
            // This means less confident weights are adjusted more.
            // This means inputs, which are zero, do not cause changes to the weights.
            INDArray adjustment = trainingSetInputs.transpose().mmul(error.mul(MathHelpers.sigmoidDerivative(output, true)));

            // Adjust the weights.
            synaptic_weights = synaptic_weights.add(adjustment);
        }
    }

    // The neural network thinks.
    INDArray think(INDArray inputs) {
        // Pass inputs through our neural network (our single neuron).
        return MathHelpers.sigmoid(inputs.mmul(synaptic_weights));
    }

    public static void main(String[] args) {

        //Initialize a single neuron neural network.
        SingleLayerNetwork neural_network = new SingleLayerNetwork();

        System.out.println("Random starting synaptic weights: ");
        System.out.println(neural_network.synaptic_weights);

        // The training set. We have 4 examples, each consisting of 3 input values
        // and 1 output value.  In this case, the first feature is the only one that matters and it dictates the result
        INDArray training_set_inputs = Nd4j.create(new double[][]{{0, 0, 1}, {1, 1, 1}, {1, 0, 1}, {0, 1, 1}});
        INDArray training_set_outputs = Nd4j.create(new double[][]{{0, 1, 1, 0}}).transpose();

        //another sample (this time we are expecting an XOR between first and second feature.  The third one is meaningless
        //this will not work with the single layer model
//        INDArray training_set_inputs = Nd4j.create(new double[][]{{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1}, {0, 0, 0}});
//        INDArray training_set_outputs = Nd4j.create(new double[][]{{0, 1, 1, 1, 1, 0, 0}}).transpose();

        // Train the neural network.
        // Do it 10,000 times and make small adjustments each time.
        neural_network.train(training_set_inputs, training_set_outputs, 10000);

        System.out.println("New synaptic weights after training: ");
        System.out.println(neural_network.synaptic_weights);

        // Test the neural network with a new situation.
        System.out.println("Considering new situation {1, 1, 0} -> ?: ");
        System.out.println(neural_network.think(Nd4j.create(new double[][]{{1, 1, 0}})));

    }
}
