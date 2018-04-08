#ifndef ML_LINEARREGRESSION_H__
#define ML_LINEARREGRESSION_H__

#include<cstddef>  // size_t;

/**
 * Simple Linear Regression,
 * Only support one feature.
 */
class LinearRegression {

	public:
		// First Feature
		double *x;
		
		// Target feature
		double *y;

		// Number of training examples
		size_t m;
		
		// the theta coefficients.	
		double *theta;
	
		/**
		 * Create a new instance from the given data set.
		 */
		LinearRegression(double x[], double y[], size_t m);

		/**
		 * Train the model with the supplied parameters.
		 *
		 * @param alpha 			The learing rate, e.g. 0.01
		 * @param iterations 	The number of gradient descent steps to do.
		 */
		void train(double alpha, int iterations);

		/**
		 * Try to predict y, given an x.
		 */
		double predict(double x);

	private:
		
		/**
		 * Compute the cost function J
		 */
		static double compute_cost(double x[], double y[], double theta[], size_t m);

		/**
		 * Compute the hypothesis.
		 */
		static double h(double x, double theta[]);

		/**
		 * Calculate the target feature from the other one.
		 */
		static double* calculate_predictions(double x[], double theta[], size_t m);

		/**
		 * Performs gradient descent to learn theta by taking num_iters gradient steps with learning rate alpha.
		 */
		static double *gradient_descent(double x[], double y[], double alpha, int iters, double *J, size_t m); 
};

#endif // ML_LINEARREGRESSION_H__
