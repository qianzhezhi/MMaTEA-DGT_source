package etmo.problems.DVA;

import etmo.core.ProblemSet;

import java.io.IOException;

public class SOProblem1 {

    public static ProblemSet getProblem() throws IOException {
        String[] riskTypes = new String[] {"infec", "symp", "hosp", "icu", "death"};
        String city = "default";
        ProblemSet problemSet = new ProblemSet(5);

        for (int i = 0; i < riskTypes.length; ++i) {
            problemSet.add(new base(city, riskTypes[i]));
            problemSet.get(i).setName("DVA_" + city + "_" + riskTypes[i]);
        }
        return problemSet;
    }
}
