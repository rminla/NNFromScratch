import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.image.DataBufferByte;
import java.util.Arrays;

public class LabeledImage {

    private byte[] pixels;
    //    private double[] pixelsNormalized;
    private int label;

    public LabeledImage(int label, byte[] pixels) {
        this.label = label;
        this.pixels = pixels;
    }

    public byte[] getPixels() {
        return pixels;
    }

    public void setPixels(byte[] pixels) {
        this.pixels = pixels;
    }

    public int getLabel() {
        return label;
    }

    public void setLabel(int label) {
        this.label = label;
    }

    public double[] getAsArrayOfDouble(byte normalizationOffset, int normalizationFactor) {
        double[] pixelsAsDouble = new double[pixels.length];
        for (int x = 0; x < pixels.length; x++) {
            pixelsAsDouble[x] = (pixels[x]) / (double) normalizationFactor;
        }
        return pixelsAsDouble;
    }
}
