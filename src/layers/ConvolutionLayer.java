/**
 * Code written by Noah Sternberg, adapted from a YouTube
 * tutorial series by Eva-Rae "Rae is Online" McLean.
 * Written 7/15/23, published 7/16/23
 */

package layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import data.MatrixUtility;

public class ConvolutionLayer extends Layer{

    private final long seed;
    private List<double[][]> filters;
    private final int filterSize;
    private final int stepSize;

    private final int inLength;
    private final int inRows;
    private final int inCols;
    private final double learningRate;

    private List<double[][]> lastInput;

    public ConvolutionLayer(int filterSize, int stepSize, int inLength, int inRows, int inCols, long SEED, int numFilters, double learningRate) {
        this.filterSize = filterSize;
        this.stepSize = stepSize;
        this.inLength = inLength;
        this.inRows = inRows;
        this.inCols = inCols;
        this.seed = SEED;
        this.learningRate = learningRate;

        generateRandomFilters(numFilters);
    }

    private void generateRandomFilters(int numFilters){
        List<double[][]> filters = new ArrayList<>();
        Random random = new Random(seed);

        for(int n = 0; n < numFilters; n+= 1) {
            double[][] newFilter = new double[filterSize][filterSize];

            for(int i = 0; i < filterSize; i+= 1){
                for(int j = 0; j < filterSize; j+= 1){

                    double value = random.nextGaussian();
                    newFilter[i][j] = value;
                }
            }
            filters.add(newFilter);
        }
        this.filters = filters;
    }

    public List<double[][]> convolutionForwardPass(List<double[][]> list){
        lastInput = list;

        List<double[][]> output = new ArrayList<>();

        for (double[][] doubles : list) {
            for (double[][] filter : filters) {
                output.add(convolve(doubles, filter, stepSize));
            }

        }
        return output;
    }

    private double[][] convolve(double[][] input, double[][] filter, int stepSize) {

        int outRows = (input.length - filter.length)/stepSize + 1;
        int outCols = (input[0].length - filter[0].length)/stepSize + 1;

        int inRows = input.length;
        int inCols = input[0].length;

        int fRows = filter.length;
        int fCols = filter[0].length;

        double[][] output = new double[outRows][outCols];

        int outRow = 0;
        int outCol;

        for(int i = 0; i <= inRows - fRows; i += stepSize){

            outCol = 0;

            for(int j = 0; j <= inCols - fCols; j+= stepSize){

                double sum = 0.0;

                for(int x = 0; x < fRows; x+= 1){
                    for(int y = 0; y < fCols; y+= 1){
                        int inputRowIndex = i+x;
                        int inputColIndex = j+y;

                        double value = filter[x][y] * input[inputRowIndex][inputColIndex];
                        sum+= value;
                    }
                }
                output[outRow][outCol] = sum;
                outCol+= 1;
            }
            outRow+= 1;
        }
        return output;
    }

    public double[][] spaceArray(double[][] input){

        if(stepSize == 1){
            return input;
        }

        int outRows = (input.length - 1) * stepSize + 1;
        int outCols = (input[0].length -1) * stepSize +1;

        double[][] output = new double[outRows][outCols];

        for(int i = 0; i < input.length; i+= 1) {
            for(int j = 0; j < input[0].length; j+= 1) {
                output[i * stepSize][j * stepSize] = input[i][j];
            }
        }
        return output;
    }


    @Override
    public double[] getOutput(List<double[][]> input) {
        return next.getOutput(convolutionForwardPass(input));
    }

    @Override
    public double[] getOutput(double[] input) {
        return getOutput(vectorToMatrix(input, inLength, inRows, inCols));
    }

    @Override
    public void backPropagation(double[] dLdO) {
        backPropagation(vectorToMatrix(dLdO, inLength, inRows, inCols));
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {

        List<double[][]> filtersDelta = new ArrayList<>();
        List<double[][]> dLdOPreviousLayer= new ArrayList<>();

        for(int f = 0; f < filters.size(); f+= 1){
            filtersDelta.add(new double[filterSize][filterSize]);
        }

        for(int i = 0; i < lastInput.size(); i+= 1){

            double[][] errorForInput = new double[inRows][inCols];

            for(int f = 0; f < filters.size(); f+= 1){

                double[][] currFilter = filters.get(f);
                double[][] error = dLdO.get(i * filters.size() + f);

                double[][] spacedError = spaceArray(error);
                double[][] dLdF = convolve(lastInput.get(i), spacedError, 1);

                double[][] delta = MatrixUtility.multiply(dLdF, learningRate*-1);
                double[][] newTotalDelta = MatrixUtility.add(filtersDelta.get(f), delta);
                filtersDelta.set(f, newTotalDelta);

                double[][] flippedError = flipArrayHorizontal(flipArrayVertical(spacedError));
                errorForInput = MatrixUtility.add(errorForInput, fullConvolve(currFilter, flippedError));
            }
            dLdOPreviousLayer.add(errorForInput);
        }

        for(int f = 0; f < filters.size(); f += 1){
            filters.set(f, MatrixUtility.add(filtersDelta.get(f), filters.get(f)));
        }

        if(last != null){
            last.backPropagation(dLdOPreviousLayer);
        }
    }

    public double[][] flipArrayHorizontal(double[][] array){
        int rows = array.length;
        int cols = array[0].length;

        double[][] output = new double[rows][cols];

        for(int i = 0; i < rows; i+= 1){
            System.arraycopy(array[i], 0, output[rows - i - 1], 0, cols);
        }

        return output;
    }

    public double[][] flipArrayVertical(double[][] array){
        int rows = array.length;
        int cols = array[0].length;

        double[][] output = new double[rows][cols];

        for(int i = 0; i < rows; i += 1){
            for(int j = 0; j < cols; j += 1){
                output[i][cols - j - 1] = array[i][j];
            }
        }
        return output;
    }

    private double[][] fullConvolve(double[][] input, double[][] filter) {

        int outRows = (input.length + filter.length) + 1;
        int outCols = (input[0].length + filter[0].length) + 1;

        int inRows = input.length;
        int inCols = input[0].length;

        int fRows = filter.length;
        int fCols = filter[0].length;

        double[][] output = new double[outRows][outCols];

        int outRow = 0;
        int outCol;

        for(int i = -fRows + 1; i < inRows; i += 1){

            outCol = 0;

            for(int j = -fCols + 1; j < inCols; j+= 1){

                double sum = 0.0;

                for(int x = 0; x < fRows; x+= 1){
                    for(int y = 0; y < fCols; y+= 1){
                        int inputRowIndex = i+x;
                        int inputColIndex = j+y;

                        if(inputRowIndex >= 0 && inputColIndex >= 0 && inputRowIndex < inRows && inputColIndex < inCols){
                            double value = filter[x][y] * input[inputRowIndex][inputColIndex];
                            sum += value;
                        }
                    }
                }

                output[outRow][outCol] = sum;
                outCol+= 1;
            }
            outRow+= 1;
        }
        return output;
    }

    @Override
    public int getOutputLength() {
        return filters.size()*inLength;
    }

    @Override
    public int getOutputRows() {
        return (inRows-filterSize)/ stepSize + 1;
    }

    @Override
    public int getOutputCols() {
        return (inCols-filterSize)/ stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return getOutputCols() * getOutputRows() * getOutputLength();
    }
}