// TrainingSession.cpp : Defines the entry point for the application.
//

#include "TrainingSession.h"
#include <torch/torch.h>

using namespace std;

int main()
{
	auto x =torch::tensor ({ 2,3 });

	std::cout << "Hello CMake." << x << std::endl;
	return 0;
}
