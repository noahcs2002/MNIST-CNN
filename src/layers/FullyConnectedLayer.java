/**
 * Code written by Noah Sternberg, adapted from a YouTube 
 * tutorial series by Eva-Rae "Rae is Online" McLean.
 * Written 7/15/23, published 7/16/23
 */
package layers;

import java.util.List;
import java.util.Random;

public class FullyConnectedLayer extends Layer{

    private final long seed;
    private final double[][] weights;
    private final int inLength;
    private final int outLength;
    private final double learningRate;
    private double[] lastZ;
    private double[] lastX;


    public FullyConnectedLayer(int inLength, int outLength, long seed, double learningRate) {
        this.inLength = inLength;
        this.outLength = outLength;
        this.seed = seed;
        this.learningRate = learningRate;

        weights = new double[inLength][outLength];
        setRandomWeights();
    }

    public double[] fullyConnectedForwardPass(double[] input){

        lastX = input;

        double[] z = new double[outLength];
        double[] out = new double[outLength];

        for(int i = 0; i < inLength; i += 1){
            for(int j = 0; j < outLength; j += 1){
                z[j] += input[i] * weights[i][j];
            }
        }

        lastZ = z;

        for(int i = 0; i < inLength; i += 1){
            for(int j = 0; j < outLength; j += 1){
                out[j] = reLu(z[j]);
            }
        }

        return out;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        return getOutput(matrixToVector(input));
    }

    @Override
    public double[] getOutput(double[] input) {
        double[] forwardPass = fullyConnectedForwardPass(input);

        if(next != null){
            return next.getOutput(forwardPass);
        }
        else {
            return forwardPass;
        }
    }

    @Override
    public void backPropagation(double[] dLdO) {

        double[] dLdX = new double[inLength];

        double dOdz;
        double dzdw;
        double dLdw;
        double dzdx;

        for(int k = 0; k < inLength; k += 1){

            double dLdXsum = 0;

            for(int j = 0; j < outLength; j += 1){

                dOdz = derivativeReLu(lastZ[j]);
                dzdw = lastX[k];
                dzdx = weights[k][j];

                dLdw = dLdO[j] * dOdz * dzdw;

                weights[k][j] -= dLdw * learningRate;

                dLdXsum += dLdO[j] * dOdz * dzdx;
            }

            dLdX[k] = dLdXsum;
        }

        if(last != null){
            last.backPropagation(dLdX);
        }
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {
        backPropagation(matrixToVector(dLdO));
    }

    @Override
    public int getOutputLength() {
        return 0;
    }

    @Override
    public int getOutputRows() {
        return 0;
    }

    @Override
    public int getOutputCols() {
        return 0;
    }

    @Override
    public int getOutputElements() {
        return outLength;
    }

    public void setRandomWeights(){
        Random random = new Random(seed);

        for(int i = 0; i < inLength; i += 1){
            for(int j =0; j < outLength; j += 1){
                weights[i][j] = random.nextGaussian();
            }
        }
    }

    public double reLu(double input){
        if(input <= 0){
            return 0;
        }
        else {
            return input;
        }
    }

    public double derivativeReLu(double input){
        if(input <= 0){
            return 0.01;
        }
        else {
            return 1;
        }
    }
}