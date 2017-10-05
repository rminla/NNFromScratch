import com.mem.common.utils.CommandLineParser;
import com.mem.common.utils.CommonHelpers;
import com.mem.common.utils.ParseHelpers;
import org.bytedeco.javacpp.Loader;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
//import org.nd4j.nativeblas.Nd4jCpu;
import org.nd4j.nativeblas.Nd4jCuda;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.security.InvalidParameterException;
import java.util.*;
import java.util.stream.Collectors;

public class Runner {

    public static void main(String[] args) throws NeuralNetwork.InvalidLayerSetupException, IOException, InterruptedException {
        CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true);
//        try {
//            Loader.load(Nd4jCuda.class);
//        } catch (Throwable e) {
//            String path = Loader.cacheResource(Nd4jCuda.class, "windows-x86_64/jnind4jcuda.dll").getPath();
//            new ProcessBuilder("C:\\dev\\tools\\deps walker\\depends.exe", path).start().waitFor();
//        }

        NeuronLayer layer1 = new NeuronLayer(1000, 784);
        NeuronLayer layer2 = new NeuronLayer(1000, 1000);
        NeuronLayer layer3 = new NeuronLayer(1000, 1000);
        NeuronLayer layer4 = new NeuronLayer(10, 1000);

        CommandLineParser commandLineParser = new CommandLineParser();
        commandLineParser.parse(args);

        // The training set. We have 7 examples, each consisting of 3 input values
        // and 1 output value.
        INDArray trainingSetInputs = Nd4j.create(new double[][]{{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1}, {0, 0, 0}});
        INDArray trainingSetOutputs = Nd4j.create(new double[][]{{0}, {1}, {1}, {1}, {1}, {0}, {0}}).transpose();

        INDArray testSetInputs = Nd4j.create(new double[][]{{1, 1, 0}});
        INDArray testSetOutputs = Nd4j.create(new double[][]{{0}}).transpose();

        String inputType;
        int numberOfIterations = 60000;
        if ("built-in-xor".equals(inputType = commandLineParser.getDoubleOption("input-type"))) {
            //takes the "XOR" value of the 1st and 2nd numbers in each array
            trainingSetInputs = Nd4j.create(new double[][]{{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1}, {0, 0, 0}});
            trainingSetOutputs = Nd4j.create(new double[][]{{0}, {1}, {1}, {1}, {1}, {0}, {0}}).transpose();
            testSetInputs = Nd4j.create(new double[][]{{1, 1, 0}});
            testSetOutputs = Nd4j.create(new double[][]{{0}}).transpose();
        } else if (inputType.equals("built-in-or")) {
            //takes the "OR" value of the 2nd and 3rd numbers in each array
            trainingSetInputs = Nd4j.create(new double[][]{{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1}, {0, 0, 0}});
            trainingSetOutputs = Nd4j.create(new double[][]{{1}, {1}, {1}, {1}, {0}, {1}, {0}}).transpose();
            testSetInputs = Nd4j.create(new double[][]{{1, 1, 0}});
            testSetOutputs = Nd4j.create(new double[][]{{1}}).transpose();
        } else if (inputType.equals("built-in-and")) {
            //takes the "AND" value of the 1st and 3rd numbers in each array
            trainingSetInputs = Nd4j.create(new double[][]{{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1}, {0, 0, 0}});
            trainingSetOutputs = Nd4j.create(new double[][]{{0}, {0}, {1}, {0}, {0}, {1}, {0}}).transpose();
            testSetInputs = Nd4j.create(new double[][]{{1, 1, 0}});
            testSetOutputs = Nd4j.create(new double[][]{{0}}).transpose();
        } else if (inputType.equals("built-in-nand")) {
            //takes the "NAND" value of the 1st and 3rd numbers in each array
            trainingSetInputs = Nd4j.create(new double[][]{{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1}, {0, 0, 0}});
            trainingSetOutputs = Nd4j.create(new double[][]{{1}, {1}, {0}, {1}, {1}, {0}, {1}}).transpose();
            testSetInputs = Nd4j.create(new double[][]{{1, 1, 0}});
            testSetOutputs = Nd4j.create(new double[][]{{1}}).transpose();
        } else if (inputType.equals("jpg-with-folder-labels")) {
            String rootFolder = "";
            if (!CommonHelpers.isEmpty(rootFolder = commandLineParser.getDoubleOption("root-image-folder"))) {
                //read the folder to ensure it contains labeled folders
                File[] labeledFolders = new File(rootFolder).listFiles(File::isDirectory);
                if (labeledFolders.length == 0) {
                    throw new InvalidParameterException("The root image folder provided must contained labeled subdirectories of images");
                }
                //read training data from disk
                List<LabeledImage> labeledImages = new ArrayList<>();
                for (File labeledFolder : labeledFolders) {
                    int label = ParseHelpers.parseInt(labeledFolder.getName(), -999999999);
                    if (label == -999999999) {
                        throw new InvalidParameterException("A labeled folder has a non-numeric value");
                    }
                    for (File imageFile : Arrays.stream(labeledFolder.listFiles(f -> f.getName().toLowerCase().endsWith(".jpg"))).limit(100).collect(Collectors.toList())) {
                        try {
                            BufferedImage bufferedImage = ImageIO.read(imageFile);
                            labeledImages.add(new LabeledImage(label, ((DataBufferByte) bufferedImage.getRaster().getDataBuffer()).getData()));
                        } catch (IOException e) {
                            throw new InvalidParameterException("Unable to read pixels from image: " + imageFile.getName());
                        }
                    }
                }
                //shuffle our data set
                Collections.shuffle(labeledImages);

                //split into train/test (for now, let's use one image for test for now.  later, this can be a parameter)
//                List<LabeledImage> trainingImages = labeledImages.subList(0, (int) Math.round(labeledImages.size() * 0.8));
//                List<LabeledImage> testImages = labeledImages.subList(trainingImages.size(), labeledImages.size());

                List<LabeledImage> trainingImages = labeledImages.subList(0, labeledImages.size() - 1);
                List<LabeledImage> testImages = labeledImages.subList(labeledImages.size() - 1, labeledImages.size());

//                List<double[]> trainingImagesAsNormalizedArrays = new LinkedList<>();
//                List<double[]> testImagesAsNormalizedArrays = new LinkedList<>();
//
//                List<double[]> trainingImageLabels = new LinkedList<>();
//                List<double[]> testImageLabels = new LinkedList<>();

                double[][] trainingImagesAsNormalizedArrays = new double[trainingImages.size()][];
                double[][] testImagesAsNormalizedArrays = new double[testImages.size()][];

                double[][] trainingImageLabels = new double[trainingImages.size()][];
                double[][] testImageLabels = new double[testImages.size()][];

                for (int x = 0; x < trainingImages.size(); x++) {
                    LabeledImage image = trainingImages.get(x);
                    trainingImagesAsNormalizedArrays[x] = image.getAsArrayOfDouble(Byte.MAX_VALUE, Byte.MAX_VALUE);
                    trainingImageLabels[x] = MathHelpers.digitToOneHotArray(image.getLabel());
                }

                for (int x = 0; x < testImages.size(); x++) {
                    LabeledImage image = trainingImages.get(x);
                    testImagesAsNormalizedArrays[x] = image.getAsArrayOfDouble(Byte.MAX_VALUE, Byte.MAX_VALUE);
                    testImageLabels[x] = MathHelpers.digitToOneHotArray(image.getLabel());
                }

                trainingSetInputs = Nd4j.create(trainingImagesAsNormalizedArrays);
                trainingSetOutputs = Nd4j.create(trainingImageLabels);

                testSetInputs = Nd4j.create(testImagesAsNormalizedArrays);
                testSetOutputs = Nd4j.create(testImageLabels);

            } else {
                throw new InvalidParameterException("A root image folder must be provided for the input type \"jpg-with-folder-labels\"");
            }
        }

        String numIterationsParameter;
        if (!CommonHelpers.isEmpty(numIterationsParameter = commandLineParser.getDoubleOption("num-iterations"))
                && ParseHelpers.parseInt(numIterationsParameter) > 0) {
            numberOfIterations = ParseHelpers.parseInt(numIterationsParameter);
        }
        System.out.println("Setting number of iterations to: " + numberOfIterations);

        // Combine the layers to create a neural network
        NeuralNetwork neuralNetwork = new NeuralNetwork(new LinkedList<>(Arrays.asList(layer1, layer2, layer3, layer4)));

        System.out.println("Generating random starting synaptic weights: ");
        neuralNetwork.print_weights();

        // Train the neural network using the training set.
        // Do it 60,000 times and make small adjustments each time.
        neuralNetwork.train(trainingSetInputs, trainingSetOutputs, numberOfIterations);

        System.out.println("Synaptic weights after training: ");
        neuralNetwork.print_weights();

        // Test the neural network with a new situation.
        System.out.println("Considering test data set: " + inputType);

        neuralNetwork.passDataThroughNetwork(testSetInputs);

        System.out.println("Test Set output: " + ND4JHelpers.getPrintableValues(testSetOutputs));
        System.out.println("Network output: " + ND4JHelpers.getPrintableValues(neuralNetwork.getResults()));
        System.out.println("Final mean error: " + MathHelpers.getMeanError(testSetOutputs, neuralNetwork.getResults()));

    }

}
