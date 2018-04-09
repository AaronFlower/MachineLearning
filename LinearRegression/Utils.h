#ifndef LINEAR_REGRESSION_UTILS_H__
#define LINEAR_REGRESSION_UTILS_H__

#include <vector>
#include <string>

using std::string;
using std::vector;

class Utils {

	public:
		static void loadData(string filename, vector<vector<double>> &features, vector<double> &lables, char delimeter = '\0');
	
	private:
		static vector<double> parseLine(string line, char delimeter);
};

#endif

