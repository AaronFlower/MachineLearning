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

void testLoadDataFromFile() {
	vector<vector<double>> features;
	vector<double> labels;

	Utils::loadData("ex1data1.txt", features, labels);

	cout << "Features : \n" << endl;
	for (auto row : features) {
		for (auto v : row) {
			cout << v << "\t" ;
		}
		cout << endl;
	}

	cout << "Labels : \n" << endl;
	for (auto v: labels) {
		cout << v << endl;
	}
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
