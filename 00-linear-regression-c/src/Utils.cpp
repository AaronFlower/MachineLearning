#include <cstddef>
#include <cassert>
#include <iostream>
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

