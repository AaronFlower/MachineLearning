#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "Utils.h"

using namespace std;

vector<double>  parse_line(string line) {
	vector<double> result;
	double value;	
	istringstream stm(line);
		
	while(stm >> value) result.push_back(value);


	return result;
}

void outputFeaturesAndLabels(const vector<vector<double>> &features, const vector<double> &labels) {
	cout << "Features :" << endl;
	for (auto row : features) {
		for (auto v : row) {
			cout << v << "\t" ;
		}
		cout << " | ";
	}

	cout << endl; 

	cout << "Labels :" << endl;
	for (auto v: labels) {
		cout << v << "\t";
	}
	cout << endl;
}

void testLoadDataFromFile() {
	vector<vector<double>> features;
	vector<double> labels;

	Utils::loadData("ex1data1.csv", features, labels, ',');
	cout << "ex1data1.csv Features and labels" <<endl;
	outputFeaturesAndLabels(features, labels);
	features.clear();
	labels.clear();

	Utils::loadData("ex1data1.txt", features, labels);
	cout << "ex1data1.txt Features and labels" <<endl;
	outputFeaturesAndLabels(features, labels);
	features.clear();
	labels.clear();

	Utils::loadData("ex1data2.csv", features, labels, ',');
	cout << "ex1data2.csv Features and labels" <<endl;
	outputFeaturesAndLabels(features, labels);
	features.clear();
	labels.clear();

	Utils::loadData("ex1data2.txt", features, labels);
	cout << "ex1data2.txt Features and labels" <<endl;
	outputFeaturesAndLabels(features, labels);
	features.clear();
	labels.clear();
}

int main () {
	string file("1 1 2 3 5 \n 8 13 21 34 55");
	vector<double> v;

	istringstream is(file);

	string line;
	while (getline(is, line)) {
		cout << line << endl;
		v = parse_line(line);
		for (auto value : v) {
			cout << value << endl;
		}
	}
	
	vector<double> features(v.begin(), v.end() - 1);
	cout << "New features \n" << endl;
	for (auto f : features) {
		cout << f << "\t";
	}
	cout << "Access iterator of vector " << endl;
	cout << *v.begin() << endl;	
	cout << *(v.end() - 1) << endl;	

	testLoadDataFromFile();
	return 0;
}
