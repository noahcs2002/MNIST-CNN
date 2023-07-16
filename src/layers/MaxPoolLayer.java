/**
 * Code written by Noah Sternberg, adapted from a YouTube
 * tutorial series by Eva-Rae "Rae is Online" McLean.
 * Written 7/15/23, published 7/16/23
 */
package layers;

import java.util.ArrayList;
import java.util.List;

public class MaxPoolLayer extends Layer{

    private final int stepSize;
    private final int windowSize;

    private final int inLength;
    private final int inRows;
    private final int inCols;

    List<int[][]> lastMaxRow;
    List<int[][]> lastMaxCol;


    public MaxPoolLayer(int _stepSize, int _windowSize, int _inLength, int _inRows, int _inCols) {
        this.stepSize = _stepSize;
        this.windowSize = _windowSize;
        this.inLength = _inLength;
        this.inRows = _inRows;
        this.inCols = _inCols;
    }

    public List<double[][]> maxPoolForwardPass(List<double[][]> input) {

        List<double[][]> output = new ArrayList<>();
        lastMaxRow = new ArrayList<>();
        lastMaxCol = new ArrayList<>();

        for (double[][] doubles : input) {
            output.add(pool(doubles));
        }

        return output;

    }

    public double[][] pool(double[][] input){

        double[][] output = new double[getOutputRows()][getOutputCols()];

        int[][] maxRows = new int[getOutputRows()][getOutputCols()];
        int[][] maxCols = new int[getOutputRows()][getOutputCols()];

        for(int r = 0; r < getOutputRows(); r+= stepSize){
            for(int c = 0; c < getOutputCols(); c+= stepSize){

                double max = 0.0;
                maxRows[r][c] = -1;
                maxCols[r][c] = -1;

                for(int x = 0; x < windowSize; x += 1){
                    for(int y = 0; y < windowSize; y += 1) {
                        if(max < input[r+x][c+y]){
                            max = input[r+x][c+y];
                            maxRows[r][c] = r+x;
                            maxCols[r][c] = c+y;
                        }
                    }
                }

                output[r][c] = max;

            }
        }

        lastMaxRow.add(maxRows);
        lastMaxCol.add(maxCols);

        return output;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        return next.getOutput(maxPoolForwardPass(input));
    }

    @Override
    public double[] getOutput(double[] input) {
        return getOutput(vectorToMatrix(input, inLength, inRows, inCols));
    }

    @Override
    public void backPropagation(double[] dLdO) {
        backPropagation(vectorToMatrix(dLdO, getOutputLength(), getOutputRows(), getOutputCols()));
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {

        List<double[][]> dXdL = new ArrayList<>();

        int l = 0;
        for(double[][] array: dLdO){
            double[][] error = new double[inRows][inCols];

            for(int r = 0; r < getOutputRows(); r += 1){
                for(int c = 0; c < getOutputCols(); c += 1){
                    int max_i = lastMaxRow.get(l)[r][c];
                    int max_j = lastMaxCol.get(l)[r][c];

                    if(max_i != -1){
                        error[max_i][max_j] += array[r][c];
                    }
                }
            }

            dXdL.add(error);
            l += 1;
        }

        if(last != null){
            last.backPropagation(dXdL);
        }
    }

    @Override
    public int getOutputLength() {
        return inLength;
    }

    @Override
    public int getOutputRows() {
        return (inRows - windowSize)/ stepSize + 1;
    }

    @Override
    public int getOutputCols() {
        return (inCols - windowSize)/ stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return inLength *getOutputCols()*getOutputRows();
    }
}