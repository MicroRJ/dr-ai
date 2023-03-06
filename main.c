#include "cc/cc.c"
#include "dr-ai.h"

// -- Note: load the sample files and train the network up to a certain threshold...
int main(int c, char **s)
{
ccdebugnone=ccfalse;
ccini();
ccdbenter("main");
	ccdebuglog("program is currently running with debug features on");
  network_t network;
  load_trained_network(&network);
ccdbleave("main");
ccdebugend();
}
