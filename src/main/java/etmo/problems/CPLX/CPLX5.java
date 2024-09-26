package etmo.problems.CPLX;

import java.io.IOException;

import  etmo.core.Problem;
import  etmo.core.ProblemSet;
import  etmo.problems.CPLX.base.*;


public class CPLX5 {
	
	public static ProblemSet getProblem() throws IOException {
		ProblemSet ps1 = getT1();
		ProblemSet ps2 = getT2();
		ProblemSet problemSet = new ProblemSet(2);

		problemSet.add(ps1.get(0));
		problemSet.add(ps2.get(0));
		return problemSet;

	}
	
	
	public static ProblemSet getT1() throws IOException {
		ProblemSet problemSet = new ProblemSet(1);
		
		MMDTLZ prob = new MMDTLZ(2, 50, 1, -100,100);
		prob.setGType("F4");

		
		double[] shiftValues = IO.readShiftValuesFromFile("resources/MData/CPLX/benchmark_5/bias_1");
		prob.setShiftValues(shiftValues);
		
		double[][] matrix = IO.readMatrixFromFile("resources/MData/CPLX/benchmark_5/matrix_1");
		prob.setRotationMatrix(matrix);	
		
		((Problem)prob).setName("CPLX5-1");
		
		problemSet.add(prob);
		return problemSet;
	}
	
	
	public static ProblemSet getT2() throws IOException {
		ProblemSet problemSet = new ProblemSet(1);
		
		
		MMZDT prob = new MMZDT(50, 1, -100,100);
		prob.setGType("F4");
		prob.setHType("concave");
		
		
		double[] shiftValues = IO.readShiftValuesFromFile("resources/MData/CPLX/benchmark_5/bias_2");
		prob.setShiftValues(shiftValues);
		
		double[][] matrix = IO.readMatrixFromFile("resources/MData/CPLX/benchmark_5/matrix_2");
		prob.setRotationMatrix(matrix);	
		
		
		((Problem)prob).setName("CPLX5-2");
		
		problemSet.add(prob);
		return problemSet;
	}
}
