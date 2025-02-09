#include "tutorialOne.h"

tutorialOne::tutorialOne()
{
	runTutorialOne();
}

void tutorialOne::runTutorialOne()
{
	/* Examples from https://github.com/AllentDan/LibtorchTutorials/tree/main/lesson2-TensorOperations  */

	auto zeroTensor = torch::zeros({ 3, 4 }); // A tensor with 3 rows and 4 colunms, this code will create  3 x 4 tensor
	std::cout << "Zero Tensor: \n" << zeroTensor << "\n"; // [ [0,0,0,0], [0,0,0,0], [0,0,0,0]]

	auto oneTensor = torch::ones({ 2, 3 }); // A tensor with 2 rows and 3 colunms, this code will create  2 x 3 tensor
	std::cout << "One Tensor: \n" << oneTensor << "\n"; // [ [1,1,1], [1,1,1]]

	/*
	* The identity matrix acts as a neutral element in matrix multiplication
	* The identity matrix is a square matrix with ones on the diagonal and zeros elsewhere.
	* They help in defining transformation matrices in computer vision
	*/
	auto eyeTensor = torch::eye(4); // A tensor with 4 rows and 4 colunms, this code will create  4 x 4 tensor.
	std::cout << "eye Tensor: \n" << eyeTensor << "\n"; // [[1,0,0,0], [0,1,0,0],[0,0,1,0],[0,0,0,1]]

	/*
	* Function full produces a tensor of the specified value and size
	*/
	auto fullTensor = torch::full({ 3,4 }, 10); // A tensor with 3 rows and 4 colunms, this code will create  3 x 4 tensor.
	std::cout << "full Tensor: \n" << fullTensor << "\n"; // [ [10,10,10,10], [10,10,10,10],[10,10,10,10],[10,10,10,10]]

	auto simpleTensor = torch::tensor({ 33,22,11 });// A tensor with 3 rows.
	std::cout << "A simple Tensor:\n" << simpleTensor << "\n"; // [33, 22, 11]


	/* ----------- Random intitialization ------------------- */
	std::cout << "----------- Random intitialization ------------------- \n";

	auto randTensor = torch::rand({ 3, 4 }); // rand generates a random value between 0 -1, A tensor with 3 rows and 4 colunms, this code will create  3 x 4 tensor.
	std::cout << "rand Tensor: \n" << randTensor << "\n"; // [[0.6400, 0.2377, 0.2625, 0.9444],[0.7461, 0.2347 , 0.1737, 0.5441],[0.5214, 0.6729, 0.7972, 0.8869]]

	auto randnTensor = torch::randn({ 2,2 }); // randn takes th random value of the normal distribution N(0,1), A tensor with 2 rows and 2 colunms, this code will create 2 X 2 tensor.
	std::cout << "randn Tensor: \n" << randnTensor << "\n"; // [[ 1.3227, 1.1869],[1.0160, -1.7203]]

	auto randIntTensor = torch::randint(0, 4, { 2,2 }); // randint takes the random integer value of min , max. A tensor with 2 rows and 2 colums, this code will create 2 x 2 tensor.
	std::cout << "randint Tensor: \n" << randIntTensor << "\n"; //[[0 , 3], [1, 3]]

	/* ---------- Convert from other data types -------------*/
	std::cout << "----------- Convert from other data types ------------------- \n";

	int intArray[3]{ 3, 4, 5 };
	//for (auto const int_array : intArray)
	//{
	//	std::cout << int_array << "\n";
	//}
	auto intArrayToTensor = torch::from_blob(intArray, { 3 }, torch::kFloat); // A tensor with 3 rows and 1 colums, this code will create 3 x 1 tensor. 
	std::cout << "IntArray: \n" << intArrayToTensor << "\n"; //[[4.2039],[5.6052],[7.0065]]

	std::vector<float> floatVector = { 6, 7, 8, 9, 10, 11 };
	//for (auto const float_vector : intArray)
	//{
	//	std::cout << float_vector << "\n";
	//}
	auto floatVectorToTensor = torch::from_blob(floatVector.data(), { 2, 3 }, torch::kFloat); // A tensor with 2 rows and 3 colums, this code will create 2 x 3 tensor. 
	std::cout << "float vector to tensor: \n" << floatVectorToTensor << "\n"; // [[6, 7, 8],[9, 10, 11]]

	/* ---------- Initializing tensor according to the existing tensor ------------*/
	std::cout << "----------- Initializing tensor according to the existing tensor ------------------- \n";

	auto firstTensor = torch::zeros({ 3, 4 });  //create a zero value tensor of 3 rows and 4 colunms (3 X 4 tensor)
	auto secondTensor = torch::Tensor(firstTensor); //create a new tensor by shallow copying firsttensor. if value of fisrttensor change then secondTensor will also change
	std::cout << "first tensor:\n" << firstTensor << "\nsecond tensor:\n" << secondTensor << "\n";

	auto zeroLikeTensor = torch::zeros_like(firstTensor); // Will generate 0 tensor with the same shape as the original tensor which is firstTensor
	auto oneLikeTensor = torch::ones_like(firstTensor); // Will generate 1 tensor with the same shape as the original tensor which is firstTensor
	auto randLikeTensor = torch::rand_like(firstTensor, torch::kFloat); // Will generate tensor with the same shape as the original tensor which is firstTensor
	std::cout << "zero Like tensor:\n" << zeroLikeTensor << "\none like tensor:\n" << oneLikeTensor << "\nrand like tensor:\n"<< randLikeTensor <<"\n";

	auto cloneTensor = firstTensor.clone(); // clone function completely copied firstTensor into a new tensor cloneTensor, the change of the original tensor firstTensor will not affect cloneTensor, which is called a deep copy.
	std::cout << "clone tensor:\n" << cloneTensor << "\n";

	/* ---------- Tensor defirmation ------------*/
	std::cout << "-----------  Tensor defirmation  ------------------- \n";

	torch::Tensor fullTensorView = torch::full({10},3);
	fullTensorView.view({1, 2, -1}); // Returns a new tensor with the same data as the self tensor but of a different shape. 
	std::cout << "full tensor as norma:\n"  << fullTensorView << "change the shape but keep the values the same:\n" << fullTensorView.view({ 1, 2, -1 }) << "\n";

	//squeeze and unspueeze


}