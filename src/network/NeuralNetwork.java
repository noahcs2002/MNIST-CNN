/**
 * Code written by Noah Sternberg, adapted from a YouTube
 * tutorial series by Eva-Rae "Rae is Online" McLean.
 * Written 7/15/23, published 7/16/23
 */
package network;

import data.Image;
import layers.Layer;

import java.util.ArrayList;
import java.util.List;

import data.MatrixUtility;

public class NeuralNetwork {

    private final List<Layer> layers;
    private final double scaleFactor;

    public NeuralNetwork(List<Layer> _layers, double scaleFactor) {
        this.layers = _layers;
        this.scaleFactor = scaleFactor;
        linkLayers();
    }

    private void linkLayers(){

        if(layers.size() <= 1){
            return;
        }

        for(int i = 0; i < layers.size(); i += 1){
            if(i == 0){
                layers.get(i).setNext(layers.get(i+1));
            }
            else if (i == layers.size()-1){
                layers.get(i).setLast(layers.get(i-1));
            }
            else {
                layers.get(i).setLast(layers.get(i-1));
                layers.get(i).setNext(layers.get(i+1));
            }
        }
    }

    public double[] getErrors(double[] networkOutput, int correctAnswer){
        int numClasses = networkOutput.length;
        double[] expected = new double[numClasses];
        expected[correctAnswer] = 1;
        return MatrixUtility.add(networkOutput, MatrixUtility.multiply(expected, -1));
    }

    private int getMaxIndex(double[] in){

        double max = 0;
        int index = 0;

        for(int i = 0; i < in.length; i += 1){
            if(in[i] >= max){
                max = in[i];
                index = i;
            }
        }

        return index;
    }

    public int guess(Image image){
        List<double[][]> inList = new ArrayList<>();
        inList.add(MatrixUtility.multiply(image.data(), (1.0 / scaleFactor)));

        double[] out = layers.get(0).getOutput(inList);

        return getMaxIndex(out);
    }

    public float test (List<Image> images){
        int correct = 0;

        for(Image img: images){
            int guess = guess(img);

            if(guess == img.label()){
                correct += 1;
            }
        }

        return ((float) correct/images.size());
    }

    public void train (List<Image> images){

        for(Image img : images){
            List<double[][]> inList = new ArrayList<>();
            inList.add(MatrixUtility.multiply(img.data(), (1f / scaleFactor)));

            double[] out = layers.get(0).getOutput(inList);
            double[] dldO = getErrors(out, img.label());

            layers.get((layers.size() - 1)).backPropagation(dldO);
        }
    }
}