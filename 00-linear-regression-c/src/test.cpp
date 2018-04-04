#include <iostream>
#include <cstddef>
#include <iomanip>      // std::setprecision
#include <cmath>
#include "utils.h"

using namespace std;

static int testCount = 0;
static int testPass = 0;
static int mainRet  = 0;

/**
 * 自已定义一个简单的测试框架。
 */
#define EXPECT_EQUAL_BASE(expect, actual)\
	do {\
		testCount++;\
		if ((expect) == (actual)) {\
			testPass++;\
		} else {\
			cerr << __FILE__ << ":" << __LINE__ << ": expect: " << (expect) << " actual: " << (actual) << endl;\
			mainRet = 1;\
		}\
	}while(0)

void testArraySum()
{
	double arr[] = {1, 2, 3, -3, -2, -1};
	size_t len = sizeof(arr) / sizeof(arr[0]);
	EXPECT_EQUAL_BASE(6, Utils::array_sum(arr, 3));
	EXPECT_EQUAL_BASE(0, Utils::array_sum(arr, len));
	EXPECT_EQUAL_BASE(0, Utils::array_sum(arr, 0));
}

void testArrayPower() 
{
	double arr[] = {1, 2, 3, -3, -2, -1};
	size_t len = sizeof(arr) / sizeof(arr[0]);
	double* pArr = Utils::array_power(arr, len, 2);
	for (size_t i = 0; i < len; ++i) {
		EXPECT_EQUAL_BASE(pow(arr[i], 2), pArr[i]);
	}
}

void testArrayMultiplication() 
{
	double arr1[] = {1, 2, 3, -3, -2, -1};
	double arr2[] = {1, 2, 3, -3, -2, -1};
	size_t len = sizeof(arr1) / sizeof(arr1[0]);
	double* pArr = Utils::array_multiplication(arr1, arr2, len);
	for (size_t i = 0; i < len; ++i) {
		EXPECT_EQUAL_BASE(arr1[i] * arr2[i], pArr[i]);
	}
}

void testArrayDiff() 
{
	double arr1[] = {1, 2, 3, -3, -2, -1};
	double arr2[] = {1, 2, 3, -3, -2, -1};
	size_t len = sizeof(arr1) / sizeof(arr1[0]);
	double* pArr = Utils::array_diff(arr1, arr2, len);
	for (size_t i = 0; i < len; ++i) {
		EXPECT_EQUAL_BASE(arr1[i] - arr2[i], pArr[i]);
	}
}
int main() {
	testArraySum();
	testArrayPower();
	testArrayMultiplication();
	testArrayDiff();
	double percentage = testPass * 100.0 / testCount;
	cout << fixed << setprecision(2);
	cout << testPass << "/" << testCount << "(" << percentage << "%) passed \n";
	return mainRet;
}
