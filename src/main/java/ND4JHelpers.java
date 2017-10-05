import org.apache.commons.lang3.StringUtils;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class ND4JHelpers {

    public static String getPrintableValues(INDArray array) {

        return getPrintableValues(array, array.rank(), 0);

    }

    final static String sep = ",";

    private static String getPrintableValues(INDArray arr, int rank, int offset) {

        StringBuilder sb = new StringBuilder();
        if (arr.isScalar()) {
            if (arr instanceof IComplexNDArray)
                return ((IComplexNDArray) arr).getComplex(0).toString();
            return String.valueOf(arr.getDouble(0));
        } else if (rank <= 0)
            return "";

        else if (arr.isVector()) {
            sb.append("[");
            for (int i = 0; i < arr.length(); i++) {
                if (arr instanceof IComplexNDArray)
                    sb.append(((IComplexNDArray) arr).getComplex(i).toString());
                else
                    sb.append(arr.getDouble(i));
                if (i < arr.length() - 1) {
                    sb.append(sep);
                    sb.append(" ");
                }
            }
            sb.append("]");
            return sb.toString();
        } else {
            offset++;
            sb.append("[");
            for (int i = 0; i < arr.slices(); i++) {
                sb.append(getPrintableValues(arr.slice(i), rank - 1, offset));
                if (i != arr.slices() - 1) {
                    sb.append(sep + " \n");
                    sb.append(StringUtils.repeat("\n", rank - 2));
                    sb.append(StringUtils.repeat(" ", offset));
                }
            }
            sb.append("]");
            return sb.toString();
        }
    }

    public static INDArray digitToOneHotINDArray(int digit) {
        double[] oneHotEncoding = new double[10];
        for (int x = 0; x <= 9; x++) {
            oneHotEncoding[x] = (x == digit) ? 1 : 0;
        }
        return Nd4j.create(oneHotEncoding);
    }
}
