#include "Utils.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using std::ifstream;
using std::istringstream;
using std::cout;
using std::endl;

void Utils::loadData(string filename, vector<vector<double>> &features, vector<double> &labels) {
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
		rowValues = parseLine(line);
		rowFeatures = vector<double>(rowValues.begin(), rowValues.end() - 1);
		cout << "Kidding: " ;
		for (auto a:rowValues) {
			cout << a << " | \t " ;
		}
		cout << endl;

		features.push_back(rowFeatures);
		labels.push_back(*(rowValues.end() - 1));
	}	
}


vector<double> Utils::parseLine(string line) {
	istringstream ism(line);
	vector<double> result;
	string strValue;
	
	while(getline(ism, strValue, ',')) {
		result.push_back(stod(strValue));
	}
	return result;
}
