import data.DataReader;
import data.Image;
import network.NetworkFactory;
import network.NeuralNetwork;
import java.time.Duration;
import java.util.List;
import java.util.Collections;
import java.util.Random;

public class Main {

    // args[0] is the seed
    public static void main(String[] args) {
        long _seed;
        Random random = new Random();

        if(args.length > 0) {
            _seed = Long.parseLong(args[0]);
        }
        
        else {
            _seed = random.nextLong();
        }

        long startTime = System.currentTimeMillis();

        System.out.println("Loading Data ...");

        List<Image> imagesTest = new DataReader().readData("data/mnist_test.csv");
        List<Image> imagesTrain = new DataReader().readData("data/mnist_train.csv");

        System.out.println("Data loaded successfully.");

        System.out.printf("Training dataframe size: %d\n",imagesTrain.size());
        System.out.printf("Testing dataframe size: %d\n",imagesTest.size());

        NetworkFactory builder = new NetworkFactory(28,28,256*100);
        builder.addConvolutionLayer(8, 5, 1, 0.1, _seed);
        builder.addMaxPoolLayer(3,2);
        builder.addFullyConnectedLayer(10, 0.1, _seed);

        NeuralNetwork net = builder.newNeuralNetwork();

        float rate = net.test(imagesTest);
        System.out.println("Pre training success rate: " + (rate*100) + "%");

        int epochs = 3;

        for(int i = 0; i < epochs; i++){
            Collections.shuffle(imagesTrain);
            net.train(imagesTrain);
            rate = net.test(imagesTest);
            System.out.println("Round " + (i + 1) + " training complete. Correct Guess Rate: " + (rate*100) + "%");
        }

        long endTime = System.currentTimeMillis();
        long elapsed = endTime - startTime;
        Duration d = Duration.ofMillis(elapsed);
        int hours = d.toHoursPart();
        int mins = d.toMinutesPart();
        int seconds = d.toSecondsPart();

        String minsString = "";
        String secondsString = "";
        if (mins < 10) {
            minsString = "0";
        }

        if(seconds < 10) {
            secondsString = "0";
        }

        minsString += mins;
        secondsString += seconds;

        System.out.printf("Elapsed time: %d:%s:%s", hours, minsString, secondsString);
    }
}