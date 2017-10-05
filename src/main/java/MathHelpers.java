import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sigmoid;
import org.nd4j.linalg.api.ops.impl.transforms.SigmoidDerivative;
import org.nd4j.linalg.factory.Nd4j;

public class MathHelpers {

    // The Sigmoid function, which describes an S shaped curve.
    // We pass the weighted sum of the inputs through this function to
    // normalise them between 0 and 1.
    public static INDArray sigmoid(INDArray matrix) {
//        return 1 / (1 + Math.exp(-x));
        return Nd4j.getExecutioner().execAndReturn(new Sigmoid(matrix));
    }

    // The derivative of the Sigmoid function.
    // This is the gradient of the Sigmoid curve.
    // It indicates how confident we are about the existing weight.
    public static INDArray sigmoidDerivative(INDArray matrix, boolean inputIsSigmoid) {
        if (!inputIsSigmoid) {
            return Nd4j.getExecutioner().execAndReturn(new SigmoidDerivative(matrix));
        } else {
            // if the input is already the result of a sigmoid, then the sigmoid derivative is simply x * (1 - x);
            INDArray ones = Nd4j.ones(matrix.shape()[0], matrix.shape()[1]);
            return matrix.mul(ones.sub(matrix));
        }
    }

    // The derivative of the Sigmoid function.
    // This is the gradient of the Sigmoid curve.
    // It indicates how confident we are about the existing weight.
    public static INDArray derivative(INDArray matrix) {
//        return x * (1 - x);
        return Nd4j.getExecutioner().execAndReturn(new Sigmoid(matrix).derivative());
    }

    public static double meanOfVector(INDArray array) {
      int arrayLength = 0;
        if (!array.isVector()) {
//            throw new UnsupportedOperationException("Cannot calculate mean on an n-dim array where n != 1");
            arrayLength = array.size(0);
        } else {
            arrayLength = array.length();
        }

        int x = 0;
        double sum = 0;
        for (x = 0; x < arrayLength; x++) {
            double value = 0d;
            if (!array.isVector()) {
//                if not a vector, try to flatten to a number
                INDArray row = array.getRow(x);
                double rowSum = 0;
                for (int rowIndex = 0; rowIndex < row.length(); rowIndex++) {
                    rowSum += Math.abs(row.getDouble(rowIndex));
                }
                value = rowSum / (double)row.length();
            } else {
                value = array.getDouble(x);
            }
            sum += Math.abs(value);
        }
        return sum / arrayLength;
    }

    public static double getMeanError(INDArray testSetOutputs, INDArray networkOutput) {

        if (!testSetOutputs.isVector() || !networkOutput.isVector()) {
            throw new UnsupportedOperationException("Mean error is only supported for vectors");
        } else if (testSetOutputs.length() != networkOutput.length()) {
            throw new UnsupportedOperationException("Mean error is only supported for vectors");
        }

        return meanOfVector(testSetOutputs.sub(networkOutput));

    }

    public static double[] digitToOneHotArray(int digit) {
        double[] oneHotEncoding = new double[10];
        for (int x = 0; x <= 9; x++) {
            oneHotEncoding[x] = (x == digit) ? 1 : 0;
        }
        return oneHotEncoding;
    }

}
