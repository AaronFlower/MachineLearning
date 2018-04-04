#include <iostream>
#include <cstddef>
#include <iomanip>      // std::setprecision
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

int main() {
	testArraySum();
	double percentage = testPass * 100.0 / testCount;
	cout << fixed << setprecision(2);
	cout << testPass << "/" << testCount << "(" << percentage << "%) passed \n";
	return mainRet;
}
