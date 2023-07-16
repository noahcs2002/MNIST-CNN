/**
 * Code written by Noah Sternberg, adapted from a YouTube
 * tutorial series by Eva-Rae "Rae is Online" McLean.
 * Written 7/15/23, published 7/16/23
 */
package network;

import layers.ConvolutionLayer;
import layers.FullyConnectedLayer;
import layers.Layer;
import layers.MaxPoolLayer;

import java.util.ArrayList;
import java.util.List;

public class NetworkFactory {

    private final int inputRows;
    private final int inputCols;
    private final double scaleFactor;
    List<Layer> layers;

    public NetworkFactory(int _inputRows, int _inputCols, double _scaleFactor) {
        this.inputRows = _inputRows;
        this.inputCols = _inputCols;
        this.scaleFactor = _scaleFactor;
        layers = new ArrayList<>();
    }

    public void addConvolutionLayer(int numFilters, int filterSize, int stepSize, double learningRate, long seed){
        if(layers.isEmpty()){
            layers.add(new ConvolutionLayer(filterSize, stepSize, 1, inputRows, inputCols, seed, numFilters, learningRate));
        }
        else {
            Layer prev = layers.get(layers.size()-1);
            layers.add(new ConvolutionLayer(filterSize, stepSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols(), seed, numFilters, learningRate));
        }
    }

    public void addMaxPoolLayer(int windowSize, int stepSize){
        if(layers.isEmpty()){
            layers.add(new MaxPoolLayer(stepSize, windowSize, 1, inputRows, inputCols));
        }
        else {
            Layer prev = layers.get(layers.size()-1);
            layers.add(new MaxPoolLayer(stepSize, windowSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols()));
        }
    }

    public void addFullyConnectedLayer(int outLength, double learningRate, long seed){
        if(layers.isEmpty()) {
            layers.add(new FullyConnectedLayer(inputCols * inputRows, outLength, seed, learningRate));
        }
        else {
            Layer prev = layers.get(layers.size()-1);
            layers.add(new FullyConnectedLayer(prev.getOutputElements(), outLength, seed, learningRate));
        }

    }

    public NeuralNetwork newNeuralNetwork(){
        return new NeuralNetwork(layers, scaleFactor);
    }
}