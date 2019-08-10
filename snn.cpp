/*
	Simple Neural Network in C++

	XOR Classification Problem

	Architecture:
		Input Layer
			2 input neuron
		Hidden Layer
			2 hidden neuron
		Output Layer
			1 output neuron
	
	Functions:
		Feedforward
			Activation Function
				Sigmoid Function: s(x) = 1 / 1 + e^-x
			Loss Function
				Squared Error: E = truth - prediction

		Backpropagation
			Optimization Algorithm
				Gradient Descent: W += Delta
*/


#include <iostream>
#include <vector>
#include <string>
using namespace std;

class Matrix
{
private:
	vector<vector<float>> m_matrix;
	unsigned int m_row;
	unsigned int m_col;

public:
	Matrix() : m_matrix(0), m_row(0), m_col(0) {}
	Matrix(const Matrix& m) : m_matrix(m.m_matrix), m_row(m.m_row), m_col(m.m_col) {}
	Matrix(const vector<vector<float>> m) : m_matrix(m), m_row(m.size()), m_col(m[0].size()) {}
	Matrix(int r, int c) : m_matrix(vector<vector<float>>(r, vector<float>(c))), m_row(r), m_col(c) {}

	int Row() { return m_row; }
	int Col() { return m_col; }

	Matrix operator + (const Matrix& rhs)
	{
		Matrix output(m_row, m_col);
		for (int i = 0; i < m_row; i++)
			for (int j = 0; j < m_col; j++)
				output.m_matrix[i][j] = this->m_matrix[i][j] + rhs.m_matrix[i][j];
		return output;
	}

	Matrix& operator += (const Matrix& rhs)
	{
		for (int i = 0; i < m_row; i++)
			for (int j = 0; j < m_col; j++)
				this->m_matrix[i][j] += rhs.m_matrix[i][j];
		return *this;
	}

	Matrix operator - (const Matrix& rhs)
	{
		Matrix output(m_row, m_col);
		for (int i = 0; i < m_row; i++)
			for (int j = 0; j < m_col; j++)
				output.m_matrix[i][j] = this->m_matrix[i][j] - rhs.m_matrix[i][j];
		return output;
	}

	Matrix& operator -= (const Matrix& rhs)
	{
		for (int i = 0; i < m_row; i++)
			for (int j = 0; j < m_col; j++)
				this->m_matrix[i][j] -= rhs.m_matrix[i][j];
		return *this;
	}

	Matrix operator * (const float& rhs)
	{
		Matrix output(m_row, m_col);
		for (int i = 0; i < m_row; i++)
			for (int j = 0; j < m_col; j++)
				output.m_matrix[i][j] *= rhs;
		return output;
	}

	Matrix operator * (const Matrix& rhs)
	{
		Matrix output(this->m_row, rhs.m_col);
		for (int i = 0; i < this->m_row; i++)
			for (int j = 0; j < rhs.m_col; j++)
				output.m_matrix[i][j] = this->m_matrix[i][j] * rhs.m_matrix[i][j];
		return output;
	}

	Matrix dot(const Matrix& rhs)
	{
		Matrix output(this->m_row, rhs.m_col);
		for (int i = 0; i < this->m_row; i++)
			for (int j = 0; j < rhs.m_col; j++)
				for (int k = 0; k < this->m_col; k++)
					output.m_matrix[i][j] += this->m_matrix[i][k] * rhs.m_matrix[k][j];

		return output;
	}

	Matrix t()
	{
		Matrix output(m_col, m_row);
		for (int i = 0; i < m_col; i++)
			for (int j = 0; j < m_row; j++)
				output.m_matrix[i][j] = m_matrix[j][i];
		return output;
	}

	int Size()
	{
		return m_matrix.size();
	}

	vector<float>& operator[] (size_t i)
	{
		return m_matrix[i];
	}


};

Matrix sigmoid(Matrix& m)
{
	Matrix output(m.Row(), m.Col());
	for (int i = 0; i < m.Row(); i++)
		for (int j = 0; j < m.Col(); j++)
			output[i][j] = 1 / (1 + expf(-m[i][j]));
	return output;
}

Matrix d_sigmoid(Matrix& m)
{
	Matrix output(m.Row(), m.Col());
	for (int i = 0; i < m.Row(); i++)
		for (int j = 0; j < m.Col(); j++)
			output[i][j] = m[i][j] * (1 - m[i][j]);
	return output;
}

Matrix MSE(Matrix& truth, Matrix& pred)
{
	Matrix output(truth.Row(), truth.Col());
	for (int i = 0; i < truth.Row(); i++)
		for (int j = 0; j < truth.Col(); j++)
			output[i][j] += powf((truth[i][j] - pred[i][j]), 2);
	return output;
}


int main()
{
	// Input Data
	Matrix X = vector<vector<float>>  { { 0, 0 },
					    { 0, 1 },
					    { 1, 0 },
					    { 1, 1 } };

	// Weight Matrices
	Matrix W = vector<vector<float>>  { { 0.35, 0.45, 0.40, 0.55 },
					    { 0.55, 0.45, 0.15, 0.55 } };

	Matrix W1 = vector<vector<float>> { { 0.35 },
					    { 0.40 },
					    { 0.45 }, 
					    { 0.50 } };
	// Output Data
	Matrix y = vector<vector<float>>  { { 0 },
					    { 1 },
					    { 1 },
					    { 0 } };

	// Training
	for (int e = 0; e < 10000; e++)
	{
		// Feedforward
		// Input Layer
		Matrix l0 = X;
		Matrix l0W = X.dot(W);	// Weighted Input
		// Hidden Layer
		Matrix l1 = sigmoid(l0W);
		Matrix l1W = l1.dot(W1);
		// Output Layer
		Matrix pred = sigmoid(l1W);

		// Backpropagation
		Matrix pred_error = y - pred;	// Calculate Cost
		Matrix pred_delta = pred_error * d_sigmoid(pred);

		Matrix l1_error = pred_delta.dot(W1.t());
		Matrix l1_delta = l1_error * d_sigmoid(l1);

		// Update Weights
		Matrix W1_delta = l1.t().dot(pred_delta);
		Matrix W_delta = X.t().dot(l1_delta);

		W1 += W1_delta;
		W += W_delta;

		for (int i = 0; i < pred.Row(); i++)
			for (int j = 0; j < pred.Col(); j++)
			{
				cout << "Pred: " << pred[i][j] << " Error: " << pred_error[i][j] << endl;
			}
	}

	return 0;
}
