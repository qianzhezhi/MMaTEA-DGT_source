package etmo.util.sorting;

import etmo.core.ProblemSet;
import etmo.core.SolutionSet;
import etmo.util.comparators.FitnessComparator;

public class SOSortiong {
    public static void sort(SolutionSet pop, ProblemSet problemSet, int taskID) {
        pop.sort(new FitnessComparator());
    }
}
