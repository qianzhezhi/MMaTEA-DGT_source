package etmo.metaheuristics.MMaTEA_DGT.models;

public abstract class AbstractDistribution {
    public abstract double[] sample();

    public abstract double[][] sample(int count); 
}
