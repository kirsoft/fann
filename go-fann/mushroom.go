package go_fann

import (
	"github.com/white-pony/go-fann"
	"fmt"
	"os"
)

func Mushroom(test_only bool) {
	const numLayers = 3
	const numNeuronsHidden = 32
	const desiredError = 0.0001
	const maxEpochs = 300
	const epochsBetweenReports = 10
	const networkFile = "mushroom_float.net"

	var ann *fann.Ann;

	// Create and Train or Read network from file
	if(!test_only) {
		fmt.Println("Creating network.")

		trainData := fann.ReadTrainFromFile("datasets/mushroom.train")
		ann = fann.CreateStandard(numLayers, []uint32{trainData.GetNumInput(), numNeuronsHidden, trainData.GetNumOutput()})

		fmt.Println("Training network.")
		ann.SetActivationFunctionHidden(fann.SIGMOID_SYMMETRIC_STEPWISE)
		ann.SetActivationFunctionOutput(fann.SIGMOID_SYMMETRIC)

		ann.TrainOnData(trainData, maxEpochs, epochsBetweenReports, desiredError)
		trainData.Destroy()
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

	// Test network
	fmt.Println("Testing network")

	testData := fann.ReadTrainFromFile("datasets/mushroom.test")

	ann.ResetMSE()

	var i uint32
	for i = 0; i < testData.Length(); i++ {
		ann.Test(testData.GetInput(i), testData.GetOutput(i))
	}

	fmt.Printf("MSE error on test data: %f\n", ann.GetMSE())
	testData.Destroy()

	// Save network to file and clean up fann data
	if(!test_only) { 
		fmt.Println("Saving network.")
		ann.Save(networkFile)
	}

	fmt.Println("Cleaning up.")
	ann.Destroy()
}
