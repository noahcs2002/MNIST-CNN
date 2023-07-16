/**
 * Code written by Noah Sternberg, adapted from a YouTube
 * tutorial series by Eva-Rae "Rae is Online" McLean.
 * Written 7/15/23, published 7/16/23
 */
package layers;

import java.util.ArrayList;
import java.util.List;

public abstract class Layer {


    protected Layer next;
    protected Layer last;

    public abstract double[] getOutput(List<double[][]> input);
    public abstract double[] getOutput(double[] input);

    public abstract void backPropagation(double[] dLdO);
    public abstract void backPropagation(List<double[][]> dLdO);

    public abstract int getOutputLength();
    public abstract int getOutputRows();
    public abstract int getOutputCols();
    public abstract int getOutputElements();

    public Layer getNext() {
        return next;
    }

    public void setNext(Layer next) {
        this.next = next;
    }

    public Layer getLast() {
        return last;
    }

    public void setLast(Layer last) {
        this.last = last;
    }


    public double[] matrixToVector(List<double[][]> input){

        int length = input.size();
        int rows = input.get(0).length;
        int cols = input.get(0)[0].length;

        double[] vector = new double[length*rows*cols];

        int i = 0;
        for (double[][] doubles : input) {
            for (int r = 0; r < rows; r += 1) {
                for (int c = 0; c < cols; c += 1) {
                    vector[i] = doubles[r][c];
                    i += 1;
                }
            }
        }

        return vector;
    }

    List<double[][]> vectorToMatrix(double[] input, int length, int rows, int cols){
        List<double[][]> out = new ArrayList<>();

        int i = 0;
        for(int l = 0; l < length; l += 1 ){
            double[][] matrix = new double[rows][cols];
            for(int r = 0; r < rows; r += 1){
                for(int c = 0; c < cols; c += 1){
                    matrix[r][c] = input[i];
                    i += 1;
                }
            }
            out.add(matrix);
        }
        return out;
    }
}