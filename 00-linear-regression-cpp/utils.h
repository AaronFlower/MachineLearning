#ifndef ML_UTILS_H__
#define ML_UTILS_H__

#include <cstddef> // for size_t

using namespace std; // to use std::size_t

class Utils {

public:
	static double array_sum(double arr[], size_t len);

	static double *array_power(double arr[], size_t len, int power);

	static double *array_multiplication(double arr1[], double arr2[], size_t len);

	static double *array_diff(double arr1[], double arr2[], size_t len);
};

void sayHi(void);

#endif

