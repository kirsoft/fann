package go_fann

import (
	"github.com/white-pony/go-fann"
	"fmt"
	"os"
)

func Robot(test_only bool) {
	const num_layers = 3
	const num_neurons_hidden = 96
	const desired_error = 0.001
	const networkFile = "robot_float.net"

	var ann *fann.Ann;

	// Creating
	if(!test_only) {
		fmt.Println("Creating network.");

		train_data := fann.ReadTrainFromFile("datasets/robot.train")

		ann = fann.CreateStandard(num_layers, []uint32{train_data.GetNumInput(), num_neurons_hidden, train_data.GetNumOutput()})

		fmt.Println("Training network.")

		ann.SetTrainingAlgorithm(fann.TRAIN_INCREMENTAL)
		ann.SetLearningMomentum(0.4)

		ann.TrainOnData(train_data, 3000, 10, desired_error)
		train_data.Destroy()
	} else {
		fmt.Println("Reading network.")
		_, err := os.Stat(networkFile)
    	if err == nil {
			ann = fann.CreateFromFile(networkFile)
		} else {
			fmt.Println("Can't open network file.")
			os.Exit(1)
		}
	}


	// Testing
	fmt.Println("Testing network")

	test_data := fann.ReadTrainFromFile("datasets/robot.test")

	ann.ResetMSE()

	var i uint32
	for i = 0; i < test_data.Length(); i++ {
		ann.Test(test_data.GetInput(i), test_data.GetOutput(i))
	}
	fmt.Printf("MSE error on test data: %f\n", ann.GetMSE())
	test_data.Destroy()


	// Saving
	if(!test_only) { 
		fmt.Println("Saving network.");
		ann.Save(networkFile)
	}


	// Cleaning up
	fmt.Println("Cleaning up.")
	ann.Destroy()
}
