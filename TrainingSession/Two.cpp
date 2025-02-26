#include "Two.h"

Two::Two()
{
	runTwo();
}

void Two::runTwo()
{
	auto device = (torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU; // set device to Cuda or CPU

	auto torchSample = torch::rand({ 2,3 }).to(device); // a tensor or 2 raws and 3 columns (2 x 3) 
	std::cout << "sample tensor:\n" << torchSample << "\n";

	auto torchMax = torchSample.max(); //maximum item in the tensor 
	std::cout << "torchMax:\n" << torchMax << "\n";

	auto torchMaxItem = torchMax.item();	//maximum item value in a tensor
	std::cout << "max item value in a tensor:\n" << torchMaxItem << "\n";

	std::cout << "sample tensor shape: " << torchSample.sizes() << "\n"; //to display the shape on a tensor

	/* --------------------Change the type of a tensor --------------------------------- */

	auto longTensor = torch::tensor({ {0,0,1} ,{1,1,1},{0,0,0} },device);
	std::cout << "Tesor shape: " << longTensor.sizes() << " Tensor type: " << longTensor.type() << "\n";

	auto longTensorToFloat = longTensor.to(torch::kFloat32);
	std::cout << "Tesor shape: " << longTensorToFloat.sizes() << " Tensor type: " << longTensorToFloat.type() << "\n";

	/* --------------------Change the shape of a tensor --------------------------------- */

	torch::Tensor shapeOne = torch::arange(784, torch::kFloat).to(device); // Create a tensor with 784 elements

	torch::Tensor shapeChanged = shapeOne.view({ 1, 28, 28 }); // chnage its shape in to 1 channel, 28 height, 28 width
	std::cout << "-- Tesor shape: " << shapeChanged.sizes() << "\nTensor type: " << shapeChanged.type() << "\n";

	/* --------------------Change the dimensions of a tensor --------------------------------- */

	auto tensorDimension = torch::rand({640,480,3 }, device); //create a tensor with dimension 640, 480, 3 (height, width, channel)
	std::cout << " tensor dimension: " << tensorDimension.sizes() << " type: "<< tensorDimension .type()<< "\n";

	auto tensorDimensionChange = tensorDimension.permute({ 2, 0, 1 }); // change the dimensions to 3, 640, 480 (channel, height, width)
	std::cout << " tensor dimension change: " << tensorDimensionChange.sizes() << " type: " << tensorDimensionChange.type() <<"ndim: " << tensorDimensionChange.ndimension()<< "\n"; // ndimensions will return 3 


}