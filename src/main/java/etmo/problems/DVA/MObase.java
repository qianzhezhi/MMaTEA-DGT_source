package etmo.problems.DVA;

import etmo.core.Problem;
import etmo.core.Solution;
import etmo.util.JMException;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;

public class MObase extends Problem {
    // problem params
    String city_;
    String[] riskTypes_;

    // socket server
//    String ip = "172.31.226.16";
//    int port = 20000;
    String ip = "localhost";
    int port = 20002;
    Socket client;
    PrintWriter out;
    BufferedReader in;

    public MObase(String city, int maxDay, String variant) {
        numberOfObjectives_ = 2;
        setNumberOfVariables(maxDay * 17);

        if (variant.equalsIgnoreCase("omicron")) {
            port = 20001;
        }

        city_ = city;
        riskTypes_ = new String[]{"infec", "death"};
    }

    @Override
    public void evaluate(Solution solution) throws JMException {
        try {
            connect();
            double obj1 = socketEvaluate(solution.getDecisionVariablesInDouble(), riskTypes_[0]);
            double obj2 = socketEvaluate(solution.getDecisionVariablesInDouble(), riskTypes_[1]);
            disconnect();

            solution.setObjective(startObjPos_, obj1);
            solution.setObjective(startObjPos_ + 1, obj2);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void dynamicEvaluate(Solution solution, int currentGeneration) throws JMException {

    }

    private void connect() throws IOException {
        client = new Socket(ip, port);
        out = new PrintWriter(client.getOutputStream(), true);
        in = new BufferedReader(new InputStreamReader(client.getInputStream()));
    }

    private void disconnect() throws IOException {
        out.println("exit");

        in.close();
        out.close();
        client.close();
    }

    private double socketEvaluate(double[] variables, String riskType) throws IOException {

        assert variables.length == numberOfVariables_;
        double res = 1e10;

        StringBuilder msg = new StringBuilder();
        msg.append(city_ + " " + riskType + " ");

        for (double variable : variables) {
            msg.append(Double.toString(variable)).append(' ');
        }
        out.println(msg);

        String received = "";
        while (received == null || received.equals("")) {
            received = in.readLine();
        }
        res = Double.parseDouble(received);

//        out.println("exit");
        return res;
    }
}
