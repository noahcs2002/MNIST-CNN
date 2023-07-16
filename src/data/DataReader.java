/**
 * Code written by Noah Sternberg, adapted from a YouTube
 * tutorial series by Eva-Rae "Rae is Online" McLean.
 * Written 7/15/23, published 7/16/23
 */

package data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public final class DataReader {

    public List<Image> readData(String path){
        List<Image> images = new ArrayList<>();

        try (BufferedReader dataReader = new BufferedReader(new FileReader(path))){

            String line;

            while((line = dataReader.readLine()) != null){
                String[] lineItems = line.split(",");

                int rows = 28;
                int cols = 28;
                double[][] data = new double[rows][cols];
                int label = Integer.parseInt(lineItems[0]);

                int i = 1;

                for(int row = 0; row < rows; row++){
                    for(int col = 0; col < cols; col++){
                        data[row][col] = (double) Integer.parseInt(lineItems[i]);
                        i++;
                    }
                }
                images.add(new Image(data, label));
            }
        }
        catch (Exception ex) {
            throw new IllegalArgumentException("File not found " + path, ex);
        }
        return images;
    }
}