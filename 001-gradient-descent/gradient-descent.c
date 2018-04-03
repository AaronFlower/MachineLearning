#include <stdio.h>

double dfx(double x) {
	return 4 * (x * x * x) - 9 * (x * x);
}

double gradientDescent(double cur_x, double alpha, double precision) {
	double c_error = precision + 1;
	double prev_x;
	
	while (c_error > precision) {
		prev_x = cur_x; 
		cur_x += -alpha * dfx(prev_x);
		c_error = prev_x > cur_x ? prev_x - cur_x : cur_x - prev_x;
	}

	return cur_x;
}


int main() {
	printf("The local minimum is %f\n", gradientDescent(6, 0.01, 0.00001));
}
