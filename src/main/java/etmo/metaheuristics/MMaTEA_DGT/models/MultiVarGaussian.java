package etmo.metaheuristics.MMaTEA_DGT.models;

import java.util.Random;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.linear.RealMatrix;

public class MultiVarGaussian extends AbstractDistribution {
    int d;
    RealMatrix means;
    RealMatrix sigma;
    RealMatrix sigmaSqrt;
    Random randomGenerator;

    MultivariateNormalDistribution model;

    public MultiVarGaussian(double[] means, double[][] sigma) {
        // assert means.length == sigma.length && sigma.length == sigma[0].length;
        // this.d = means.length;
        // this.means = new Array2DRowRealMatrix(means);
        // this.sigma = new Array2DRowRealMatrix(sigma);
        // this.sigmaSqrt = sqrt(this.sigma);
        // randomGenerator = new Random();
        model = new MultivariateNormalDistribution(means, sigma);
    }

    public double[] sample() {
        // return sigmaSqrt.multiply(getRandomVector()).add(means).transpose().getData()[0];
        return model.sample();
    }

    public double[][] sample(int count) {
        // double[][] samples = new double[count][d];
        // Arrays.parallelSetAll(samples, i -> getRandomVector());
        // return new Array2DRowRealMatrix(samples).multiply(sigmaSqrt).add(means).getData();
        return model.sample(count);
    }

    // public RealMatrix getRandomVector() {
    //     double[] vector = new double[d];
    //     Arrays.parallelSetAll(vector, i -> randomGenerator.nextGaussian());
    //     return new Array2DRowRealMatrix(vector);
    // }

    // RealMatrix sqrt(RealMatrix mat) {
    //     SingularValueDecomposition svd = new SingularValueDecomposition(mat);
    //     RealMatrix U = svd.getU();
    //     RealMatrix S = svd.getS();
    //     RealMatrix VT = svd.getVT();
    //     for (int i = 0; i < this.d; i ++) 
    //         S.setEntry(i, i, Math.sqrt(S.getEntry(i, i)));

    //     return U.multiply(S).multiply(VT);
    // }
}
