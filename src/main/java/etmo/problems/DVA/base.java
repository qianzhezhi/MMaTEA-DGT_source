package etmo.problems.DVA;

import etmo.core.Problem;
import etmo.core.Solution;
import etmo.util.JMException;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;

public class base extends Problem {
    // problem params
    String city_;
    String riskType_;

    // socket server
    String ip = "localhost";
    int port = 20000;
    Socket client;
    PrintWriter out;
    BufferedReader in;

    public base(String city, String riskType) {
        numberOfObjectives_ = 1;
        setNumberOfVariables(400 * 17);

        city_ = city;
        riskType_ = riskType;
    }

    @Override
    public void evaluate(Solution solution) throws JMException {
        try {
            connect();
            double result = socketEvaluate(solution.getDecisionVariablesInDouble());
            disconnect();
            solution.setFitness(result);
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
        in.close();
        out.close();
        client.close();
    }

    private double socketEvaluate(double[] variables) throws IOException {

        assert variables.length == numberOfVariables_;
        double res = 1e10;
        StringBuilder msg = new StringBuilder();
        msg.append(city_ + " " + riskType_ + " ");
        for (double variable : variables) {
            msg.append(Double.toString(variable)).append(' ');
        }
        out.println(msg);

        String received = "";
        while (received.equals("")) {
            received = in.readLine();
        }
        res = Double.parseDouble(received);

        out.println("exit");
        return res;
    }
}
