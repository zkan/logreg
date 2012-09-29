#include "logreg.h"

LogisticRegression::LogisticRegression() {
}

LogisticRegression::~LogisticRegression() {
}

// credit: uvts_cvs's reply at
// http://stackoverflow.com/questions/392981/how-can-i-convert-string-to-double-in-c
double LogisticRegression::string_to_double(string str) {
    istringstream i(str);
    double val;
    
    if(!(i >> val)) {
        return 0;
    }

    return val;
}

// credit: heeen's reply at 
// http://stackoverflow.com/questions/599989/is-there-a-built-in-way-to-split-strings-in-c
vector<string> LogisticRegression::split(string str, string delimiters) {
    vector<string> tokens;
    
    // skip delimiters at beginning
    string::size_type last_pos = str.find_first_not_of(delimiters, 0);
    // find first "non-delimiter"
    string::size_type pos = str.find_first_of(delimiters, last_pos);

    // insert first column for data X
    tokens.push_back("1");

    // npos is a static member constant value (-1) with the greatest 
    // possible value for an element of type size_t.
    // credit: http://www.cplusplus.com/reference/string/string/npos/
    while(string::npos != pos || string::npos != last_pos) {
        // found a token, add it to the vector
        tokens.push_back(str.substr(last_pos, pos - last_pos));
        // skip delimiters
        last_pos = str.find_first_not_of(delimiters, pos);
        // find next "non-delimiter"
        pos = str.find_first_of(delimiters, last_pos);
    }

    return tokens;
}

void LogisticRegression::print_data() {
    for(unsigned int i = 0; i < this->_data.size(); i++) {
        for(unsigned int j = 0; j < this->_data[i].size(); j++) {
            cout << this->_data[i][j] << "|";
        }
        cout << " => " << this->_predicted_data[i] << endl;
    }
}

void LogisticRegression::read_training_data(char* file_data) {
    ifstream ifile;
    ifile.open(file_data);

    vector<string> tokens;
    vector<double> val;
    string line;
    while(!ifile.eof()) {
        getline(ifile, line);
        // skip an empty line
        if(line.empty()) {
            continue;
        }
       
        // split the data into tokens and add 1 to the first column
        tokens = split(line, ", ");
        // first to second last column are data X
        for(unsigned int i = 0; i < tokens.size() - 1; i++) {
            val.push_back(string_to_double(tokens[i]));
        }
        this->_data.push_back(val);

        // last column is the predicted data y
        this->_predicted_data.push_back(string_to_double(tokens[tokens.size() - 1]));

        val.clear();
    }
}

double LogisticRegression::dot_product(vector<double> a, vector<double> b) {
    double sum_product = 0;

    for(unsigned int i = 0; i < a.size(); i++) {
        sum_product += a[i] * b[i];
    }

    return sum_product;
}

void LogisticRegression::feature_normalize() {
    // initialize the mean and std vectors
    for(unsigned int i = 0; i < this->_data[0].size(); i++) {
        this->_mean.push_back(0);
        this->_std.push_back(0);
    }

    // compute mean and std (both have same vector size)
    unsigned int N = this->_data.size();
    for(unsigned int i = 0; i < this->_data.size(); i++) {
        for(unsigned int j = 0; j < this->_data[i].size(); j++) {
            this->_mean[j] += this->_data[i][j];
            this->_std[j] += this->_data[i][j] * this->_data[i][j];
        }
    }
    for(unsigned int i = 0; i < this->_mean.size(); i++) {
        this->_mean[i] = this->_mean[i] / N;
        this->_std[i] = sqrt((this->_std[i] / N) - (this->_mean[i] * this->_mean[i]));
    }

    for(unsigned int i = 0; i < this->_data.size(); i++) {
        // normalize all except the first column (the first column is used for theta_0)
        for(unsigned int j = 1; j < this->_data[i].size(); j++) {
            this->_data[i][j] = (this->_data[i][j] - this->_mean[j]) / this->_std[j];
        }
    }

}

/**********************************************************************/
/*** sigmoid.c:  This code contains the function routine            ***/
/***             sigmoid() which performs the unipolar sigmoid      ***/
/***             function for backpropagation neural computation.   ***/
/***             Accepts the input value x then returns it's        ***/
/***             sigmoid value in float.                            ***/
/***                                                                ***/
/***  function usage:                                               ***/
/***       float sigmoid(float x);                                  ***/
/***           x:  Input value                                      ***/
/***                                                                ***/
/***  Written by:  Kiyoshi Kawaguchi                                ***/
/***               Electrical and Computer Engineering              ***/
/***               University of Texas at El Paso                   ***/
/***  Last update:  09/28/99  for version 2.0 of BP-XOR program     ***/
/**********************************************************************/
double LogisticRegression::sigmoid(double x) {
    double exp_value;
    double return_value;

    /*** Exponential calculation ***/
    exp_value = exp(-x);

    /*** Final sigmoid value ***/
    return_value = 1 / (1 + exp_value);

    return return_value;
}

double LogisticRegression::compute_cost(vector< vector<double> > X, 
                                      vector<double> y, 
                                      vector<double> theta) {
    double J = 0;
    unsigned int m = y.size();
    double h = 0;

    for(unsigned int i = 0; i < X.size(); i++) {
        h = sigmoid(dot_product(X[i], theta));
        J += (y[i] * log(h)) + ((1.0 - y[i]) * log(1.0 - h));
    }

    J = (-1.0 / m) * J;

    return J;
}

void LogisticRegression::gradient_descent(double alpha, int num_iters, bool norm) {
    vector<double> J_history;
    unsigned int m = this->_predicted_data.size();

    // normalize the data
    if(norm) {
        feature_normalize();
    }

    // initialize theta
    for(unsigned int i = 0; i < this->_data[0].size(); i++) {
        this->_theta.push_back(0);
    }
    cout << "Cost at initial theta (zeros): " << compute_cost(this->_data, 
                                                               this->_predicted_data, 
                                                               this->_theta) << endl
                                                               << endl;

    double J = 0;
    for(int iter = 0; iter < num_iters; iter++) {
        cout << "Iter " << iter + 1 << ": ";

        // then compute the error and update the theta
        double error = 0;
        double h = 0;
        double grad = 0;
        for(unsigned int i = 0; i < this->_theta.size(); i++) {
            vector<double> diff;
            vector<double> X_col;
            for(unsigned int j = 0; j < this->_data.size(); j++) {
                double val = dot_product(this->_data[j], this->_theta);
                h = sigmoid(val);

//                cout << "diff: " << h - this->_predicted_data[j] << endl;

                diff.push_back(h - this->_predicted_data[j]);
                X_col.push_back(this->_data[j][i]);
            }
            error = dot_product(diff, X_col);
            grad = alpha * (1.0 / m) * error;

            // update the theta
            this->_theta[i] = this->_theta[i] - grad;
        
        }

        if(iter == 0) {
            cout << "Gradient at initial theta (zeros): " << endl;
            for(unsigned k = 0; k < this->_theta.size(); k++) {
                cout << grad << endl;
            }
            cout << endl;
        }

        J = compute_cost(this->_data, this->_predicted_data, this->_theta);
        cout << J << endl;
        J_history.push_back(J);

    }
    cout << endl;
}

void LogisticRegression::classify(vector<double> X) {
    double h =  sigmoid(dot_product(X, this->_theta));
    
    cout << "h = " << h << endl;
    if(h > 0.5) {
        cout << 1 << endl;
    }
    else {
        cout << 0 << endl;
    }
}

void LogisticRegression::print_theta() {
    cout << "Theta:" << endl;
    for(unsigned int i = 0; i < this->_theta.size(); i++) {
        cout << this->_theta[i] << endl;
    }
    cout << endl;
}

vector<double> LogisticRegression::get_theta() {
    return this->_theta;
}

vector<double> LogisticRegression::get_mean() {
    return this->_mean;
}

vector<double> LogisticRegression::get_std() {
    return this->_std;
}

