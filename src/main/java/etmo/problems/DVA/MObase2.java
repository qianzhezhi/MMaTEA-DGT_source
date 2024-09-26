package etmo.problems.DVA;

import etmo.core.Problem;
import etmo.core.Solution;
import etmo.util.JMException;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;

// 精简版：
// 年龄组：17 -> 9
// 优化周期： 180 -> ceil(180 / 7)

public class MObase2 extends Problem {
    // problem params
    String city_;

    // socket server
//    String ip = "172.31.226.16";
//    int port = 20000;
    String ip = "localhost";
    int port = 20002;
    Socket client;
    PrintWriter out;
    BufferedReader in;

    public MObase2(String city, int maxDay, String variant) {
        numberOfObjectives_ = 2;
        setNumberOfVariables(((int) Math.ceil(maxDay / 7.)) * 9);

        if (variant.equalsIgnoreCase("omicron")) {
            port = 20001;
        }

        city_ = city;
    }

    @Override
    public void evaluate(Solution solution) throws JMException {
        try {
            connect();
            double[] objs = socketEvaluate(solution.getDecisionVariablesInDouble());
            disconnect();

            solution.setObjective(startObjPos_, objs[0]);
            solution.setObjective(startObjPos_ + 1, objs[1]);
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

    private double[] socketEvaluate(double[] variables) throws IOException {

        assert variables.length == numberOfVariables_;
        double[] res = new double[2];

        StringBuilder msg = new StringBuilder();
        StringBuilder tmp = new StringBuilder();
        msg.append(city_ + " ");

        for (int i = 0; i < variables.length; ++i) {
            tmp.append(variables[i]).append(' ');
            if (i % 9 < 8) tmp.append(variables[i]).append(' ');
            if (i % 9 == 8) {
                for (int j = 0; j < 7 && msg.length() < 17 * 180; ++j) {
                    msg.append(tmp);
                }
                tmp = new StringBuilder();
            }
        }

        out.println(msg);

        String received = "";
        while (received == null || received.equals("")) {
            received = in.readLine();
        }
        received = received.strip();
        String[] split = received.split(" ");
        for (int i = 0; i < 2; ++i) res[i] = Double.parseDouble(split[i]);

        return res;
    }
}
