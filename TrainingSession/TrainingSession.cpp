// TrainingSession.cpp : Defines the entry point for the application.
#include "TrainingSession.h"

using namespace std;

int main()
{
	/* Examples from https://github.com/AllentDan/LibtorchTutorials/tree/main/lesson2-TensorOperations  */

	auto zeroTensor = torch::zeros({ 3, 4 }); // A tensor with 3 rows and 4 colunms, this code will create  3 x 4 tensor
	std::cout << "Zero Tensor: \n" << zeroTensor <<std::endl; // [ [0,0,0,0], [0,0,0,0], [0,0,0,0]]

	auto oneTensor = torch::ones({ 2, 3 }); // A tensor with 2 rows and 3 colunms, this code will create  2 x 3 tensor
	std::cout << "One Tensor: \n"<< oneTensor << std::endl; // [ [1,1,1], [1,1,1]]

	/*
	* The identity matrix acts as a neutral element in matrix multiplication
	* The identity matrix is a square matrix with ones on the diagonal and zeros elsewhere.
	* They help in defining transformation matrices in computer vision
	*/
	auto eyeTensor = torch::eye(4); // A tensor with 4 rows and 4 colunms, this code will create  4 x 4 tensor.
	std::cout << "eye Tensor: \n" << eyeTensor << std::endl; // [ [1,0,0,0], [0,1,0,0],[0,0,1,0],[0,0,0,1]]

	/* 
	* Function full produces a tensor of the specified value and size
	*/
	auto fullTensor = torch::full({3,4}, 10); // A tensor with 3 rows and 4 colunms, this code will create  3 x 4 tensor.
	std::cout << "full Tensor: \n" << fullTensor << std::endl;// [ [10,10,10,10], [10,10,10,10],[10,10,10,10],[10,10,10,10]]

	auto simpleTensor = torch::tensor({ 33,22,11 });// A tensor with 3 rows.
	std::cout << "A simple Tensor:\n" << simpleTensor << std::endl; // [33, 22, 11]

	return 0;
}
