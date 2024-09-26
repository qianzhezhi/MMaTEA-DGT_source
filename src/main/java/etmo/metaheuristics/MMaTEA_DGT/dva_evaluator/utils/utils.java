package etmo.metaheuristics.MMaTEA_DGT.dva_evaluator.utils;

public class utils {
    public static byte[] variablesToBytes(double[] variables) {
        StringBuilder msgStr = new StringBuilder();
        for (int i = 0; i < variables.length; ++i) {
            msgStr.append(variables[i]);
            if (i < variables.length - 1) {
                msgStr.append(' ');
            }
        }
        byte[] buf = new byte[msgStr.length()];
        for (int i = 0; i < msgStr.length(); ++i) {
            buf[i] = (byte) msgStr.charAt(i);
        }
        return buf;
    }
}
