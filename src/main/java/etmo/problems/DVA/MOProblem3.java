package etmo.problems.DVA;

import etmo.core.ProblemSet;

import java.io.IOException;

// Delta 180 // 7 * 9

public class MOProblem3 {

    public static ProblemSet getProblem() throws IOException {
        String[] cities = new String[]{
                "default", "Beijing", "Shanghai", "Guangzhou",
                "Shenzhen", "Hangzhou", "Chengdu", "Xian",
                "Nanjing", "Wuhan", "Chongqing"
        };
        int n = cities.length;
        ProblemSet problemSet = new ProblemSet(n);
        for (int i = 0; i < n; ++i) {
            problemSet.add(new MObase2(cities[i], 180, "delta"));
            problemSet.get(i).setName("DVA_" + cities[i]);
            problemSet.get(i).setHType("convex");
        }
        return problemSet;
    }
}
