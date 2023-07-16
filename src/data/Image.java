/**
 * Code written by Noah Sternberg, adapted from a YouTube
 * tutorial series by Eva-Rae "Rae is Online" McLean.
 * Written 7/15/23, published 7/16/23
 */
package data;

public record Image(double[][] data, int label) {

    @Override
    public String toString() {

        StringBuilder s = new StringBuilder(label + ", \n");

        for (double[] datum : data) {
            for (int j = 0; j < data[0].length; j++) {
                s.append(datum[j]).append(", ");
            }
            s.append("\n");
        }

        return s.toString();
    }
}
