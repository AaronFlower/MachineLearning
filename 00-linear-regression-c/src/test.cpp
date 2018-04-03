#include <iostream>
#include <cstddef>
#include "utils.h"

using namespace std;

void testArraySum()
{
	double arr[] = {1, 2, 3};
	cout << Utils::array_sum(arr, 10);
 // 	cout << Utils::array_sum(NULL, 10);
}
int main() {
	cout << "Hello world\n" ;
	sayHi();	
	testArraySum();
	return 0;
}
