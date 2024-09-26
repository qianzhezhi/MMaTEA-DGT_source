package etmo.metaheuristics.MMaTEA_DGT;

import etmo.core.*;
import etmo.metaheuristics.MMaTEA_DGT.models.AbstractDistribution;
import etmo.util.Configuration;
import etmo.util.JMException;
import etmo.util.PseudoRandom;
import etmo.util.math.*;
import etmo.util.sorting.NDSortiong;
import etmo.util.sorting.SOSortiong;

import java.io.*;
import java.util.Arrays;

public class MaTMY3_Gaussian_SO extends MtoAlgorithm {
    private SolutionSet[] population;
    private Solution[][] offspring;
    private SolutionSet[] union;

    private int generation;
    private int evaluations;
    private int maxEvaluations;

    private int populationSize;
    private int taskNum;
    private int varNum;
    private int[] objStart;
    private int[] objEnd;

    private int[] transferTotalCount;

    private String XType;
    private String TXType;
    private Operator DECrossover;
    private Operator SBXCrossover;
    private Operator mutation;

    private int[] stuckTimes;
    private double[] bestDistances;
    private double[] previousBestDistances;

    private double mutationProbability;
    private double transferProbability;
    private double[] tP;
    private double[][] distances;
    private double[][] distances2;
    private double[][] confidences;
    private int[][] transferredCounts;
    private int[][] transferredEliteCounts;
    private double[][] lastTransferSuccessRate;

    private double elitePart;

    boolean isMutate;

    double[][] means;
    double[][] stds;
//    double[][][] sigmas;
    AbstractDistribution[] models;
    boolean[] isSingular;
    double[][] eliteDirections;

    // parallel runner
    Object[] runner;
    final Object lock = new Object();

    // DEBUG
    int selectVariableID = 1;

    // DEBUG: IGD
    boolean isProcessLog;
    double[][] processIGD;

    public MaTMY3_Gaussian_SO(ProblemSet problemSet) {
        super(problemSet);
    }

    @Override
    public SolutionSet[] execute() throws JMException, ClassNotFoundException {
        initState();
        if (isProcessLog)
            updateProcessIGD();
        while (evaluations < maxEvaluations) {
            iterate();
            // long startTime = System.currentTimeMillis();
            if (isProcessLog)
                updateProcessIGD();
            resetFlag();
        }
        // if (isPlot)
        // endPlot();
        // System.out.println(igdPlotValues.get(plotTaskID).toString());

        if (isProcessLog)
            writeProcessIGD();

        return population;
    }

    void initState() throws JMException, ClassNotFoundException {
        generation = 0;
        evaluations = 0;
        taskNum = problemSet_.size();
        varNum = problemSet_.getMaxDimension();

        maxEvaluations = (Integer) this.getInputParameter("maxEvaluations");
        populationSize = (Integer) this.getInputParameter("populationSize");

        // // DEBUG partly run
        // maxEvaluations = maxEvaluations / taskNum * 2;
        // // problemSet_ = problemSet_.getTask(0);
        // taskNum = 2;

        XType = (String) this.getInputParameter("XType");
        TXType = (String) this.getInputParameter("TXType");
        isMutate = (Boolean) this.getInputParameter("isMutate");
        isProcessLog = (Boolean) this.getInputParameter("isProcessLog");
        transferProbability = (Double) this.getInputParameter("transferProbability");
        mutationProbability = (Double) this.getInputParameter("mutationProbability");
        
        elitePart = (Double) this.getInputParameter("elitePartition");

        tP = new double[taskNum];
        Arrays.fill(tP, transferProbability);

        DECrossover = operators_.get("DECrossover");
        SBXCrossover = operators_.get("SBXCrossover");
        mutation = operators_.get("mutation");

        objStart = new int[taskNum];
        objEnd = new int[taskNum];
        bestDistances = new double[taskNum];
        previousBestDistances = new double[taskNum];
        stuckTimes = new int[taskNum];
        Arrays.fill(bestDistances, Double.MAX_VALUE);
        Arrays.fill(previousBestDistances, Double.MAX_VALUE);
        Arrays.fill(stuckTimes, 0);
        
        confidences = new double[taskNum][taskNum];
        transferredCounts = new int[taskNum][taskNum];
        lastTransferSuccessRate = new double[taskNum][taskNum];
        transferredEliteCounts = new int[taskNum][taskNum];

        distances = new double[taskNum][taskNum];
        distances2 = new double[taskNum][taskNum];

        means = new double[taskNum][varNum];
        stds = new double[taskNum][varNum];
//        sigmas = new double[taskNum][varNum][varNum];
        models = new AbstractDistribution[taskNum];
        isSingular = new boolean[taskNum];
        eliteDirections = new double[taskNum][varNum];

        population = new SolutionSet[taskNum];
        union = new SolutionSet[taskNum];
        offspring = new Solution[taskNum][populationSize];
        for (int k = 0; k < taskNum; k++) {
            objStart[k] = problemSet_.get(k).getStartObjPos();
            objEnd[k] = problemSet_.get(k).getEndObjPos();

            Arrays.fill(distances[k], 0);
            Arrays.fill(distances2[k], 0);
            Arrays.fill(means[k], 0);
            Arrays.fill(stds[k], 0);
            Arrays.fill(eliteDirections[k], 0);

            Arrays.fill(confidences[k], 0.01);
            Arrays.fill(transferredCounts[k], 0);
            Arrays.fill(lastTransferSuccessRate[k], 0.0);

            population[k] = new SolutionSet(populationSize);
            union[k] = new SolutionSet();
            for (int i = 0; i < populationSize; i++) {
                // if (k == 0) {
                Solution solution = new Solution(problemSet_);
                solution.setSkillFactor(k);
                evaluate(solution, k);
                population[k].add(solution);
                // } else {
                //     Solution solution = new Solution(population[0].get(i));
                //     solution.setSkillFactor(k);
                //     evaluate(solution, k);
                //     population[k].add(solution);
                // }
            }
            NDSortiong.sort(population[k], problemSet_, k);
            for (int i = 0; i < populationSize; i ++) {
                population[k].get(i).setFlag2(population[k].get(i).getRank());
            }
            // updateBestDistances(k);
        }

        runner = new Object[taskNum];

        processIGD = new double[taskNum][maxEvaluations/populationSize/taskNum];

        transferTotalCount = new int[taskNum];
        Arrays.fill(transferTotalCount, 0);
    }

    void iterate() throws JMException, ClassNotFoundException {
        // long time = System.currentTimeMillis();
        offspringGeneration();
        // System.out.println("offspringGeneration: " + (System.currentTimeMillis() - time));
        // time = System.currentTimeMillis();
        environmentSelection();
        // System.out.println("environmentSelection: " + (System.currentTimeMillis() - time));
        // writePopulationVariablesMatrix(plotTaskID, generation);
        generation++;
    }

    void offspringGeneration() throws JMException {
        updateDistributions(elitePart);
        updateDistances();

        // parallel
        Arrays.parallelSetAll(runner, k -> {
            Object res = null;
            try {
                res = generatingOffspring(k);
            } catch (JMException e) {
                e.printStackTrace();
            }
            return res;
        });

        // for (int k = 0; k < taskNum; k++) {
        //     Arrays.fill(offspring[k], null);
        //     Solution child;

        //     int[] perm = PseudoRandom.randomPermutation(population[k].size(), population[k].size());
        //     for (int i = 0; i < populationSize; i ++) {
        //         child = null;
        //         // if (PseudoRandom.randDouble() < transferProbability) {
        //         //     int k2 = getAssistTaskID(k);
        //         //     child = transferGenerating(k, k2, perm[i], TXType);
        //         // } else {
        //         //     child = evolutionaryGenerating(k, perm[i], XType);
        //         // }

        //         evaluate(child, k);
        //         offspring[k][i] = child;
        //     }
        // }
    }

    Object generatingOffspring(int taskID) throws JMException {
        Arrays.fill(offspring[taskID], null);
        int[] perm = PseudoRandom.randomPermutation(population[taskID].size(), population[taskID].size());
        
        Solution child;
        for (int i = 0; i < populationSize; i ++) {
            child = null;
            if (PseudoRandom.randDouble() < tP[taskID]) {
                int k2 = getAssistTaskID(taskID);
            // if (i < populationSize / 2) {
            //     int k2 = i;
                child = transferGenerating(taskID, k2, perm[i], TXType);
            //  int[] assistTaskList = getAssistTaskIDs(taskID, 10);
            //  double[] 
            } else {
                child = evolutionaryGenerating(taskID, perm[i], XType);
            }
            evaluate(child, taskID);
            child.setFlag2(-1);
            offspring[taskID][i] = child;
        }
        return null;
    }

    Solution transferGenerating(int taskID, int assistTaskID, int i, String type) throws JMException {
        int j = PseudoRandom.randInt(0, population[assistTaskID].size() - 1);
        Solution child = null;
        
        // HGT
        child = new Solution(population[assistTaskID].get(j));
        double[] tmpMean = population[assistTaskID].getMean();
        Vector.vecSub_(tmpMean, means[assistTaskID]);
        Vector.vecAdd_(tmpMean, means[taskID]);
        double[] tmpStd = population[assistTaskID].getStd();
        double[] newFeatures = Probability.sampleByNorm(tmpMean, tmpStd);

        // // MGT
        // child = new Solution(population[assistTaskID].get(j));
        // double[] tmpMean = population[taskID].getMean();
        // double[] tmpStd = population[assistTaskID].getStd();
        // double[] newFeatures = Probability.sampleByNorm(tmpMean, tmpStd);
        
        // double[][] tmpSigma = Matrix.getMatSigma(population[assistTaskID].getMat());
        // double[] newFeatures = Probability.sampleByNorm(tmpMean, tmpSigma);

        Vector.vecClip_(newFeatures, 0.0, 1.0);
        child.setDecisionVariables(newFeatures);
        mutateIndividual(taskID, child);
        
        
        // Normal Implicit
        // child = SBXChildGenerating(taskID, assistTaskID, i);
        // mutateIndividual(taskID, child);

        child.resetObjective();
        child.setFlag(1);
        return child;
    }

    Solution transferGenerating2(int taskID, int assistTaskID, int i, String type) throws JMException {
        int j = PseudoRandom.randInt(0, population[assistTaskID].size() - 1);
        
        Solution child = null;
        child = new Solution(population[assistTaskID].get(j));

        double[] newFeatures = Utils.subspaceAlignment(
            population[assistTaskID].getMat(), 
            population[taskID].getMat(),
            population[assistTaskID].get(j).getDecisionVariablesInDouble());

        Vector.vecClip_(newFeatures, 0.0, 1.0);
        child.setDecisionVariables(newFeatures);
        mutateIndividual(taskID, child);
        child.resetObjective();

        child.setFlag(1);
        return child;
    }

    Solution evolutionaryGenerating(int taskID, int i, String type) throws JMException {
        Solution child = null;
        if (type.equalsIgnoreCase("SBX")) {
            child = SBXChildGenerating(taskID, taskID, i);
        } else if (type.equalsIgnoreCase("DE")) {
            child = DEChildGenerating(taskID, taskID, i);
        }
        else {
            System.out.println("Error: unsupported reproduce type: " + type);
            System.exit(1);
        }
        child.setFlag(2);
        return child;
    }

    void environmentSelection() throws ClassNotFoundException, JMException {
        // TODO: failed to parallelize
        // Arrays.parallelSetAll(runner,  i -> environmentSelection(i));
        
        for (int taskID = 0; taskID < taskNum; taskID ++) {
            SolutionSet offspringSet = new SolutionSet(offspring[taskID]);
            SolutionSet union = population[taskID].union(offspringSet);
            SOSortiong.sort(union, problemSet_, taskID);
    
            // double improvement = 0;
            // for (int i = 0; i < union.size(); i ++) {
            //     if (union.get(i).getFlag2() >= 0) {
            //         improvement += (union.get(i).getRank() - union.get(i).getFlag2());
            //     }
            // }
            // improvement /= populationSize;
            // tP[taskID] = improvement > 1 ? 0.5 : 0.5 * (2 - improvement);
            // tP[taskID] = improvement > 1 ? 0.5 : 0.5 * improvement;
            // if (taskID == plotTaskID) {
            //     System.out.println(generation + ": " + improvement);
            //     System.out.println(generation + ": " + tP[taskID]);
            // }
            // Arrays.fill(transferredEliteCounts[taskID], 0);
            // int eliteCount = 0;
            // int transferredEliteCount = 0;
            for (int i = 0; i < populationSize; i++) {
                // if (union.get(i).getRank() <= 1) {
                // if (i < populationSize / 2) {
                //     eliteCount ++;
                //     int sf = union.get(i).getSkillFactor();
                //     if (sf != taskID) {
                //         transferredEliteCounts[taskID][sf] = 1;
                //         transferredEliteCount ++;
                //     }
                // }
                union.get(i).setFlag2(union.get(i).getRank());
                union.get(i).setSkillFactor(taskID);
                population[taskID].replace(i, union.get(i));
            }

            // tP[taskID] = 0.9 * tP[taskID] + 0.1 * (transferredEliteCount * 1.0 / eliteCount / tP[taskID]);
            // // if (taskID == plotTaskID) {
            // //     System.out.println(tP[taskID]);
            // //     System.out.println(generation + ": " + transferredEliteCount + " / " + eliteCount);
            // //     System.out.println(Arrays.toString(transferredEliteCounts[taskID]));
            // // }
            // tP[taskID] = Math.max(0.1, tP[taskID]);
            // tP[taskID] = Math.min(0.9, tP[taskID]);

            // updateBestDistances(taskID);
        }
    }

    Object environmentSelection(int taskID) {
        // SolutionSet offspringSet = new SolutionSet(offspring[taskID]);
        // SolutionSet union = population[taskID].union(offspringSet);
        union[taskID] = population[taskID].union(offspring[taskID]);
        NDSortiong.sort(union[taskID], problemSet_, taskID);

        synchronized(lock) {
            for (int i = 0; i < populationSize; i++) {
                union[taskID].get(i).setSkillFactor(taskID);
                population[taskID].replace(i, union[taskID].get(i));
            }
        }

        return null;
    }

    Solution DEChildGenerating(int taskID, int assistTaskID, int i) throws JMException {
        int j1 = i, j2 = i;
        while (j1 == i && j1 == j2) {
            j1 = PseudoRandom.randInt(0, populationSize - 1);
            j2 = PseudoRandom.randInt(0, populationSize - 1);
        }
        Solution[] parents = new Solution[3];
        parents[0] = population[assistTaskID].get(j1);
        parents[1] = population[assistTaskID].get(j2);
        parents[2] = population[taskID].get(i);

        Solution child = (Solution) DECrossover.execute(new Object[] { population[taskID].get(i), parents });
        mutateIndividual(taskID, child);

        return child;
    }

    Solution SBXChildGenerating(int taskID, int assistTaskID, int i) throws JMException {
        int j = i;
        while (j == i)
            j = PseudoRandom.randInt(0, populationSize - 1);
        Solution[] parents = new Solution[2];
        parents[0] = population[taskID].get(i);
        parents[1] = population[assistTaskID].get(j);

        Solution child = ((Solution[]) SBXCrossover.execute(parents))[PseudoRandom.randInt(0, 1)];
        mutateIndividual(taskID, child);

        return child;
    }

    void mutateIndividual(int taskID, Solution individual) throws JMException {
        // if (isMutate)
        //     mutation.execute(individual);

        // if (PseudoRandom.randDouble() < stuckTimes[taskID] * 0.15)
        //     mutation.execute(individual);

        if (PseudoRandom.randDouble() < mutationProbability) {
            mutation.execute(individual);
            // individual.setFlag(1);
        }
    }

    void updateDistributions(double partition) throws JMException {
        // double[] weights = null;
        SolutionSet[] tmpSet = new SolutionSet[taskNum];
        for (int k = 0; k < taskNum; k++) {
             int size = (int)(population[k].size() * partition);
            //  weights = new double[size];
             tmpSet[k] = new SolutionSet(size);
             for (int i = 0; i < size; i++) {
                 tmpSet[k].add(population[k].get(i));
                //  weights[i] = 1 / (population[k].get(i).getRank() + 1.0);
             }

        //    tmpSet[k] = new SolutionSet();
        //    for (int i = 0; i < population[k].size(); i ++) {
        //        if ((population[k].get(i).getRank() == 0 || tmpSet[k].size() < 10) && i < 50) {
        //            tmpSet[k].add(population[k].get(i));
        //        }
        //        else {
        //            break;
        //        }
        //    }

            // means[k] = tmpSet[k].getWeightedMean(weights);
            // stds[k] = tmpSet[k].getWeightedStd(weights);
            // means[k] = tmpSet[k].getMean();
            // sigmas[k] = Matrix.getMatSigma(tmpSet[k].getMat());
        }
        Arrays.parallelSetAll(means, k -> tmpSet[k].getMean());
        Arrays.parallelSetAll(stds, k -> tmpSet[k].getStd());
//        Arrays.parallelSetAll(sigmas, k -> {
//            double[][] output = null;
//            try {
//				output = Matrix.getMatSigma(population[k].getMat());
//			} catch (JMException e) {
//				e.printStackTrace();
//			}
//            return output;
//		});
        // Arrays.setAll(eliteDirections, k -> Vector.vecSub(means[k], population[k].getMean()));
    }

    int[] getAssistTaskIDs(int taskID, int num) {
        int[] idx = new int[num];
        Arrays.fill(idx, -1);
        for (int k = 0; k < taskNum; k++) {
            if (k == taskID) continue;
            for (int i = 0; i < num; i ++) {
                if (idx[i] == -1) {
                    idx[i] = k;
                }
                else if (distances[taskID][idx[i]] > distances[taskID][k]) {
                    for (int j = i + 1; j < num; j++) {
                        if (idx[j] == -1) {
                            idx[j] = idx[j-1];
                            break;
                        }
                        idx[j] = idx[j-1];
                    }
                    idx[i] = k;
                }
            }
        }
        return idx;
    }

    int getAssistTaskID(int taskID) throws JMException {
        int assistTaskID = taskID;

        // // random
        // while (assistTaskID == taskID) {
        //     assistTaskID = PseudoRandom.randInt(0, taskNum - 1);
        // }

        // double[] scores = new double[taskNum];
        // Arrays.setAll(scores, i -> transferredEliteCounts[taskID][i]);
        // assistTaskID = Random.rouletteWheel(scores, taskID);

        // double[] scores = new double[taskNum];
        // Arrays.setAll(scores, i -> distances[taskID][i]);
        // assistTaskID = Random.rouletteWheel(scores, taskID);

        // CMD + EMD
        double[] scores = new double[taskNum];
        // CMD
        Arrays.setAll(scores, i -> distances[taskID][i]);
        int res1 = Random.rouletteWheel(scores, taskID);
        // EMD
        Arrays.setAll(scores, i -> 1 / distances2[taskID][i]);
        int res2 = Random.rouletteWheel(scores, taskID);
        assistTaskID = PseudoRandom.randDouble() < 0.5 ? res1 : res2;

        // // CMD + EMD 2
        // double[] scores = new double[taskNum];
        // Arrays.setAll(scores, i -> distances[taskID][i] / distances[taskID][i]);
        // assistTaskID = Random.rouletteWheel(scores, taskID);

        // int length = 10;
        // double[] scores = new double[length];
        // while (assistTaskID == taskID) {
        //     int[] perm = PseudoRandom.randomPermutation(taskNum, length);
        //     Arrays.setAll(scores, i -> distances[taskID][perm[i]]);
        //     assistTaskID = perm[Random.rouletteWheel(scores)];
        // }

        // // coral distance
        // int[] perm = PseudoRandom.randomPermutation(taskNum, taskNum);
        // double minDist = Double.MAX_VALUE;
        // int selectTaskID = -1;
        // // TODO: hard code
        // for (int k = 0; k < 10; k ++) {
        //     if (distances[taskID][k] < minDist) {
        //         minDist = distances[taskID][perm[k]];
        //         selectTaskID = perm[k];
        //     }
        // }
        // assistTaskID = selectTaskID == -1 ? taskID : selectTaskID;
        
        // assistTaskID = Random.rouletteWheel(distances[taskID], taskID);

        return assistTaskID;
    }
  
    void updateDistances() throws JMException {
        for (int k = 0; k < taskNum; k++) {
           final int srcTaskID = k;
           Arrays.parallelSetAll(distances[k], trgTaskID -> {
                double dist = 0;
                if (trgTaskID > srcTaskID) {
                        // int d = sigmas[srcTaskID].length;
                        // for (int i = 0; i < d; i ++) {
                        //     for (int j = 0; j < d; j ++) {
                        //         dist += Math.pow(sigmas[srcTaskID][i][j] - sigmas[trgTaskID][i][j], 2);
                        //     }
                        // }
                        // dist = Math.sqrt(dist);
                        dist = Distance.getDistance(stds[srcTaskID], stds[trgTaskID]);
//                        dist = Distance.getCorrelationMatrixDistance(sigmas[srcTaskID], sigmas[trgTaskID]);
                        // dist = Distance.getCosineSimilarity(eliteDirections[srcTaskID], eliteDirections[trgTaskID]);
                        // dist = (1 + dist) / 2 * -1;
                    } else {
                        dist = distances[trgTaskID][srcTaskID];
                    }
                return dist;
            });
            Arrays.parallelSetAll(distances2[k], trgTaskID -> {
            double dist = 0;
            if (trgTaskID > srcTaskID) {
                    // int d = sigmas[srcTaskID].length;
                    // for (int i = 0; i < d; i ++) {
                    //     for (int j = 0; j < d; j ++) {
                    //         dist += Math.pow(sigmas[srcTaskID][i][j] - sigmas[trgTaskID][i][j], 2);
                    //     }
                    // }
                    // dist = Math.sqrt(dist);
                    dist = Distance.getDistance(means[srcTaskID], means[trgTaskID]);
                } else {
                    dist = distances2[trgTaskID][srcTaskID];
                }
            return dist;
            });
        }
    }
      
    void updateBestDistances(int taskID) {
        // boolean updated = false;
        double avgDistance = 0;
        for (int j = 0; j < population[taskID].size(); j++) {
            double distance = 0;
            for (int i = objStart[taskID]; i <= objEnd[taskID]; i++) {
                distance += Math.pow(population[taskID].get(j).getObjective(i), 2);
            }
            distance = Math.sqrt(distance);
            avgDistance += distance;
        }
        avgDistance /= population[taskID].size();

        if (generation >= 2) {
            double improve1 = Math.max(0, previousBestDistances[taskID] - bestDistances[taskID]);
            double improve2 = Math.max(0, bestDistances[taskID] - avgDistance);
            tP[taskID] = improve2 / (improve1 + improve2 + 1e-13);
        }

        previousBestDistances[taskID] = bestDistances[taskID];
        bestDistances[taskID] = avgDistance;

        
        // if (Math.abs(avgDistance - bestDistances[taskID]) > bestDistances[taskID] * 5e-3) {
        //     if (avgDistance < bestDistances[taskID]) {
        //         updated = true;
        //     }
        //     bestDistances[taskID] = avgDistance;
        // }
            
        // if (updated) {
        //     stuckTimes[taskID] = 0;
        // } else {
        //     stuckTimes[taskID]++;
        // }
    }

    void evaluate(SolutionSet population, int taskID) {

    }

    void evaluate(Solution solution, int taskID) throws JMException {
        synchronized (lock){
            // solution.setSkillFactor(taskID);
            problemSet_.get(taskID).evaluate(solution);
            evaluations ++;
        }
    }

    void catastrophe(int taskID, double survivalRate, int threshold) throws ClassNotFoundException, JMException {
        if (stuckTimes[taskID] >= threshold
                && evaluations < maxEvaluations - 100) {
            // System.out.println(evaluations + ": task " + taskID +" : reset on " + selectVariableID);
            int[] perm = PseudoRandom.randomPermutation(populationSize, (int) (populationSize * (1 - survivalRate)));
            double[] ms = population[taskID].getMean();
            double[] ss = population[taskID].getStd();
            for (int i = 0; i < perm.length; i++) {
                // // totally random individual
                // Solution solution = new Solution(problemSet_);
                // solution.setSkillFactor(taskID);
                // problemSet_.get(taskID).evaluate(solution);
                // // evaluations++;
                // population[taskID].replace(perm[i], solution);
                
                // partily random vaiable
                Vector.vecElemMul_(ss, selectVariableID);
                Vector.vecClip_(ss, 0.0, 0.5);
                Solution tmp = population[taskID].get(perm[i]);
                double[] newFeatures = Probability.sampleByNorm(ms, ss);
                Vector.vecClip_(newFeatures, 0.0, 1.0);
                tmp.setDecisionVariables(newFeatures);
                problemSet_.get(taskID).evaluate(tmp);
                population[taskID].replace(perm[i], tmp);
            }

            stuckTimes[taskID] = 0;
        }
    }

    void resetFlag() {
        for (int k = 0; k < taskNum; k++) {
            for (int i = 0; i < population[k].size(); i++) {
                population[k].get(i).setFlag(0);
            }
        }
    }

    void writePopulationVariablesMatrix(int taskID, int generation) throws JMException {
        String algoName = "Gaussian_exp"; 
        String folderPath = "./data/variables";
        File folder = new File(folderPath);
        if (!folder.exists()){
            folder.mkdirs();
        }
        String filePath = folderPath + "/" + algoName + "_" + generation + ".txt";
        double[][] data = population[taskID].getMat();
        try {
            FileOutputStream fos = new FileOutputStream(filePath);
            OutputStreamWriter osw = new OutputStreamWriter(fos);
            BufferedWriter bw = new BufferedWriter(osw);

            for (double[] line: data) {
                String sLine = Arrays.toString(line)
                    .replace("[", "")
                    .replace("]", "")
                    .replace(",", "")
                    .strip();
                bw.write(sLine);
                bw.newLine();
            }
            bw.close();
        } catch (IOException e) {
            Configuration.logger_.severe("Error acceding to the file");
            e.printStackTrace();
        }

    }

    void writeProcessIGD() throws JMException {
        // 就在data目录创建，具体的目录结构由上层函数再处理
        String folderPath = "./data";
  
        String filePath = folderPath + "/" + "tmp.txt";
        double[][] data = processIGD;
        try {
            FileOutputStream fos = new FileOutputStream(filePath);
            OutputStreamWriter osw = new OutputStreamWriter(fos);
            BufferedWriter bw = new BufferedWriter(osw);

            for (double[] line: data) {
                String sLine = Arrays.toString(line)
                    .replace("[", "")
                    .replace("]", "")
                    .replace(",", "")
                    .strip();
                bw.write(sLine);
                bw.newLine();
            }
            bw.close();
        } catch (IOException e) {
            Configuration.logger_.severe("Error acceding to the file");
            e.printStackTrace();
        }
    }

    void updateProcessIGD() {
        for (int k = 0; k < taskNum; k++) {
            processIGD[k][generation] = population[k].get(0).getFitness();
            System.out.println(generation + " - " + (k + 1) + ": " + processIGD[k][generation]);
        }
    }

}
