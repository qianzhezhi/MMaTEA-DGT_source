package etmo.metaheuristics.MMaTEA_DGT.dva_evaluator;

import java.io.*;
import java.util.Arrays;
import java.util.Random;

import etmo.core.ProblemSet;
import etmo.core.Solution;
import etmo.core.SolutionSet;
import etmo.problems.DVA.MOProblem1;
import etmo.util.JMException;

public class client {
    public static void main(String[] args) throws IOException, ClassNotFoundException, JMException {
        String ip = "localhost";
        int port = 20001;

        int len = 17 * 400;
        double[] variables = new double[len];
        Random r = new Random();
        for (int i = 0; i < len; ++i) {
            variables[i] = r.nextDouble();
        }

        StringBuilder msg = new StringBuilder();
        msg.append("default infec ");
        for (double variable : variables) {
            msg.append(Double.toString(variable));
            msg.append(' ');
        }

        long tStart = System.currentTimeMillis();

        int populationSize = 10;
        ProblemSet ps = MOProblem1.getProblem();
        SolutionSet population = new SolutionSet(populationSize);
        Object[] runner = new Object[populationSize];

        Arrays.parallelSetAll(runner, i -> {
            try {
//                Socket socket = new Socket(ip, port);
//                if (socket.isConnected()) {
//                    System.out.println(i + " ready");
//                    PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
//                    BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
//                    out.println("aaaaaaaaaaaaaaaaaaaa hello by: " + i);
//                    out.println("bbbbbbbbbbbbbbbbbbbb hello by: " + i);
//                    String rec = in.readLine();
//                    System.out.println(rec);
//                } else {
//                    System.out.println(i + " not ready");
//                }
//                socket.close();

                Solution solution = null;
                solution = new Solution(ps);
                ps.get(0).evaluate(solution);
                population.add(solution);
                System.out.println("done: " + i);
            } catch (ClassNotFoundException | JMException e) {
                e.printStackTrace();
            }

            return null;
        });

//        for (int i = 0; i < populationSize; i++) {
//            // if (k == 0) {
//            Solution solution = new Solution(ps);
//            ps.get(0).evaluate(solution);
//            population.add(solution);
//            System.out.println("done: " + i);
//        }

//        double[] result = new double[50];
//
//        for (int i = 0; i < 50; ++i) {
//            Socket client = new Socket(ip, port);
//            PrintWriter out = new PrintWriter(client.getOutputStream(), true);
//            BufferedReader in = new BufferedReader(new InputStreamReader(client.getInputStream()));
//            out.println(msg);
//
//            String received = "";
//            while (received.equals("")) {
//                received = in.readLine();
//            }
//            double res = Double.parseDouble(received);
//            System.out.println(res);
//            result[i] = res;
//        }

        System.out.println("Running cost: " + (System.currentTimeMillis() - tStart) + " ms.");
    }
}
