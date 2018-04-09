#include "Utils.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using std::ifstream;
using std::istringstream;
using std::cout;
using std::endl;

void Utils::loadData(string filename, vector<vector<double>> &features, vector<double> &labels, char delimeter) {
	ifstream isRead(filename);	
	
	if (!isRead.is_open()) {
		cout << "Error occurred in opening " << filename << endl;
	} else {
		cout << "Loading " << filename << " data ... " << endl;
	}


	string line;
	vector<double> rowValues;
	vector<double> rowFeatures;

	while(getline(isRead, line)) {
		rowValues = parseLine(line, delimeter);
		features.push_back(vector<double>(rowValues.begin(), rowValues.end() - 1));
		labels.push_back(*(rowValues.end() - 1));
	}	
}


vector<double> Utils::parseLine(string line, char delimeter) {
	istringstream ism(line);
	vector<double> result;
	string strValue;
	double value;
	if (delimeter == '\0') {
		while (ism >> value) result.push_back(value);
	} else {
		while (getline(ism, strValue, delimeter)) result.push_back(stod(strValue));
	}	

	return result;
}
