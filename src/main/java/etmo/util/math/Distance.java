package etmo.util.math;
import java.util.Arrays;

public class Distance {
    public static double getDistance(double[] p1, double[] p2) {
        return getDistance(p1, p2, 2);
    }

    public static double getDistance(double[] p1, double[] p2, int d) {
        double distance = 0;
        for (int i = 0; i < p1.length; i++) {
            distance += Math.abs(Math.pow(p1[i] - p2[i], d));
        }
        return Math.pow(distance, 1.0 / d);
    }

    public static double getCosineSimilarity(double[] v1, double[] v2) {
        double similarity = 0;
        similarity = Vector.vecDot(v1, v2) / (Vector.vecModule(v1) * Vector.vecModule(v2));
        return similarity;
    }

    public static double getWassersteinDistance(double[] p1, double[] p2) {
// 二维数组情况
//        double distance = 0;
//        int size = p1.length;
//        double[][] distanceMat = new double[size][size];
//        for (int i = 0; i < size - 1; i++){
//            distanceMat[i][i] = Double.MAX_VALUE;
//            for (int j = i + 1; j < size; j++){
//                distanceMat[i][j] = distanceMat[j][i] = getDistance(p1[i], p2[j]);
//            }
//        }
//
//        int[] assignment = new HungarianAlgorithm(distanceMat).execute();
//        for (int i = 0; i < size; i++){
//            distance += distanceMat[i][assignment[i]];
//        }
//
//        return distance / size;
        //一维数组情况
        // Sort arrays to compute CDF
        Arrays.sort(p1);
        Arrays.sort(p2);

        int n = p1.length;
        double wassersteinDist = 0.0;

        // Compute Wasserstein distance
        for (int i = 0; i < n; i++) {
            wassersteinDist += Math.abs(p1[i] - p2[i]);
        }

        return 1-wassersteinDist;
    }

    public static double getCoralLoss(double[][] p1, double[][] p2) {
        int d = p1[0].length; 
        double dist = 0;
        double[][] sigma1 = Matrix.getMatSigma(p1);
        double[][] sigma2 = Matrix.getMatSigma(p2);

        for (int i = 0; i < d; i ++) {
            for (int j = 0; j < d; j ++) {
                dist += Math.pow(sigma1[i][j] - sigma2[i][j], 2);
            }
        }
        return Math.sqrt(dist) / (4.0 * Math.pow(d, 2));
    }

    public static double getCoralLossWithSigma(double[][] sigma1, double[][] sigma2) {
        int d = sigma1.length; 
        double dist = 0;
        for (int i = 0; i < d; i ++) {
            for (int j = 0; j < d; j ++) {
                dist += Math.pow(sigma1[i][j] - sigma2[i][j], 2);
            }
        }
        return Math.sqrt(dist) / (4.0 * Math.pow(d, 2));
    }

    public static double getCorrelationMatrixDistance(double[][] sigma1, double[][] sigma2) {
        double dist = 0;
        dist = Matrix.matTrace(Matrix.matMul(sigma1, sigma2)) / Matrix.matNorm(sigma1) / Matrix.matNorm(sigma2);
        return 1 - dist;
    }

    public static double getKLDivergence(double[] p1, double[] p2) {
        double dist = 0;
        // TODO 二维数组时,记得换输入参数
//        for (int i = 0; i < p1.length; i++) {
//            for (int j = 0; j < p1[0].length; j++) {
//                // 处理概率值为0的情况
//                if (p1[i][j] == 0 || p2[i][j] == 0) {
//                    continue; // 忽略该项
//                }
//
//                dist += p1[i][j] * Math.log(p1[i][j] / p2[i][j]);
//            }
//        }
        // Check if dimensions match
        if (p1.length != p2.length) {
            throw new IllegalArgumentException("Arrays must have the same length.");
        }

        int length = p1.length;

        // Compute KL divergence
        for (int i = 0; i < length; i++) {
            if (p1[i] != 0 && p2[i] != 0) { // Ensure neither is zero to avoid NaN in log calculation
                dist += p1[i] * Math.log(p1[i] / p2[i]);
            }
        }
        return 1-dist;
    }

    public static double getJSDivergence(double[] p1, double[] p2) {
       // double dist = 0;
        // TODO
//        if (p1.length != p2.length || p1[0].length != p2[0].length) {
//            throw new IllegalArgumentException("输入二维数组维度不一致");
//        }
//
//        double[][] average = new double[p1.length][p1[0].length];
//        for (int i = 0; i < p1.length; i++) {
//            for (int j = 0; j < p1[0].length; j++) {
//                average[i][j] = (p1[i][j] + p2[i][j]) / 2.0;
//            }
//        }
        // Compute average distribution M
        double[] M = new double[p1.length];
        for (int i = 0; i < p1.length; i++) {
            M[i] = (p1[i] + p2[i]) / 2.0;
        }

        // Compute JS divergence
        double dist = (getKLDivergence(p1, M) + getKLDivergence(p2, M)) / 2.0;

        return dist;
    }
}
