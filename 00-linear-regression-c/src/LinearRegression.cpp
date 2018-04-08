#include <cstddef> // size_t
#include <iostream>
#include "utils.h"
#include "LinearRegression.h"

using namespace std;

LinearRegression::LinearRegression(double x[], double y[], size_t m) {
	this->x = x;
	this->y = y;
	this->m = m;
}

double LinearRegression::h(double x, double theta[]) {
	return theta[0] + theta[1];
}

double LinearRegression::predict(double x) {
	return h(x, theta);
}

double* LinearRegression::calculate_predictions(double x[], double theta[], size_t m) {
	double* predictions = new double[m];

	for (size_t i = 0; i < m; ++i) {
		predictions[i] = h(x[i], theta);
	}

	return predictions;
}

double LinearRegression::compute_cost(double x[], double y[], double theta[], size_t m) {
	double* predictions = calculate_predictions(x, theta, m);
	double* diff = Utils::array_diff(predictions, y, m);
	double* sq_errors = Utils::array_power(diff, m, 2);
	return (1.0 / (2 * m)) * Utils::array_sum(sq_errors, m);
}

double *LinearRegression::gradient_descent(double x[], double y[], double alpha, int iters, double *J, size_t m) {
	double *theta = new double[2];
	theta[0] = 1;
	theta[1] = 1;

	for (size_t i = 0; i < iters; ++i) {
		double *predictions = calculate_predictions(x, theta, m);
		double *diff = Utils::array_diff(predictions, y, m);

		/**
		 * because the features x1 value is all 1.
		 */
		double *errors_x1 = diff;
		double *errors_x2 = Utils::array_multiplication(diff, x, m);

		theta[0] = theta[0] - alpha * (1.0 / m) * Utils::array_sum(errors_x1, m);
		theta[1] = theta[1] - alpha * (1.0 / m) * Utils::array_sum(errors_x2, m);

		J[i] = compute_cost(x, y, theta, m);
	}
	
	return theta;
}

void LinearRegression::train(double alpha, int iters) {
	double *J = new double[iters];

	this->theta = gradient_descent(x, y, alpha, iters, J, m);
	
	cout << "J = ";
	for (size_t i =0; i < iters; ++i) {
		cout << J[i] << ' '; 
	}
	cout << endl << "Theta: " << theta[0] << " " << theta[1] << endl;
}

