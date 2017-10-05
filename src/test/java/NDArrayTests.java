import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sigmoid;
import org.nd4j.linalg.api.ops.impl.transforms.SigmoidDerivative;
import org.nd4j.linalg.factory.Nd4j;

import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;

public class NDArrayTests {

    @Test
    public void testSigmoidOnTwoDimensionalMatrix() {
        float[][] values = new float[][]
                {
                        {0, 0, 1},
                        {0, 1, 1},
                        {1, 0, 1},
                        {0, 1, 0},
                        {1, 0, 0},
                        {1, 1, 1},
                        {0, 0, 0}
                };
        INDArray matrix = Nd4j.create(values);
        INDArray transformedMatrix = sigmoid(matrix);
        System.out.println(transformedMatrix);
    }
    @Test
    public void testSigmoidDerivativeOnTwoDimensionalMatrix() {
        float[][] values = new float[][]
                {
                        {0, 0, 1},
                        {0, 1, 1},
                        {1, 0, 1},
                        {0, 1, 0},
                        {1, 0, 0},
                        {1, 1, 1},
                        {0, 0, 0}
                };
        INDArray matrix = Nd4j.create(values);
        INDArray transformedMatrix = Nd4j.getExecutioner().execAndReturn(new Sigmoid(matrix).derivative());
        INDArray transformedMatrix2 = Nd4j.getExecutioner().execAndReturn(new SigmoidDerivative(matrix));
        System.out.println(transformedMatrix);
        System.out.println(transformedMatrix2);


    }

    @Test
    public void testSigmoid() {
        Sigmoid sigmoid = new Sigmoid();
    }
}
