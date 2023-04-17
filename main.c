#include "cc/cc.c"
#include "dr-ai.h"

// note: to download the dataset, it's the same website and everything,
// but these links point to the specific files...
// https://www.kaggle.com/datasets/hojjatk/mnist-dataset?resource=download&select=train-images-idx3-ubyte
// https://www.kaggle.com/datasets/hojjatk/mnist-dataset?resource=download&select=train-labels-idx1-ubyte
// Unzip the files and put the files under the data directory.

// -- Note: load the sample files and train the network up to a certain threshold...
int main(int c, char **s)
{
ccinit();
ccdbenter("main");
	ccdebuglog("program is currently running with debug features on");
  network_t network;
  load_trained_network(&network);
ccdbleave("main");
ccdebugend();
}
