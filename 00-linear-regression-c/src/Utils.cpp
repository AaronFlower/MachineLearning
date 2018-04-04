#include <cstddef>
#include <cassert>
#include <iostream>
#include <cmath>
#include "utils.h"

using namespace std;

void sayHi(void) {
	cout << "Hi" <<endl;
}


double Utils::array_sum(double arr[], size_t len)
{
	assert(arr != NULL);

	double sum = 0;
	
	for (size_t i = 0; i < len; ++i) {
		sum += arr[i];		
	}

	return sum;
}

double *Utils::array_power(double arr[], size_t len, int power)
{
	assert(arr != NULL);
	
	double* newArr = new double[len];
	for(size_t i = 0; i < len; ++i) {
		newArr[i] = pow(arr[i], power);
	}	
	
	return newArr;
}

double* Utils::array_multiplication(double arr1[], double arr2[], size_t len) 
{
	assert(arr1 != NULL && arr2 != NULL);

	double* newArr = new double[len];
	for(size_t i = 0; i < len; ++i) {
		newArr[i] = arr1[i] * arr2[i];
	}

	return newArr;
}

double* Utils::array_diff(double arr1[], double arr2[], size_t len)
{
	assert(arr1 != NULL && arr2 != NULL);

	double* newArr = new double[len];
	for(size_t i = 0; i < len; ++i) {
		newArr[i] = arr1[i] - arr2[i];
	}

	return newArr;
}
