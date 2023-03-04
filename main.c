#include "..\\cc\\cc.c"
#include "dr-ai.h"

// --- Note: add to cc.c
ccfunc ccinle unsigned int
ccu32_reverse(unsigned int i)
{ return (((i & 0xFF000000) >> 0x18) |
          ((i & 0x000000FF) << 0x18) |
          ((i & 0x00FF0000) >> 0x08) |
          ((i & 0x0000FF00) << 0x08));
}

// --- Note: from stb!
void printimage(int w, int h, unsigned char *image)
{
  char v[2] = "";
  for (int y=0;y<h;++y)
  { for (int x=0;x<w;++x)
    { int q=image[w*y+x];
      *v=" .:ioVM@"[q>>5];
      printf(v);
    }
    printf("\n");
  }
}

// -- Note: load the sample files and train the network up to a certain threshold...
int main(int c, char **s)
{
ccdebugnone=ccfalse;
ccini();

ccdbenter("main");

  void
    *labels_file=ccopenfile("data\\train-labels.idx1-ubyte","r"),
    *images_file=ccopenfile("data\\train-images.idx3-ubyte","r");

  if(!labels_file)
  {
    cctraceerr("could not find labels file");
  }

  if(!images_file)
  {
    cctraceerr("could not find images file");
  }

  if(!labels_file || !images_file)
  {
    return 7;
  }

  unsigned char *labels=ccpullfile(labels_file,0,0);
  unsigned char *images=ccpullfile(images_file,0,0);

  if(!labels)
  {
    cctraceerr("could not read labels file");
  }

  if(!images)
  {
    cctraceerr("could not read images file");
  }

  if(!labels || !images)
  {
    return 7;
  }

  unsigned int image_size;
  int image_count;
  int image_size_x;
  int image_size_y;

  int total_images;
  int total_labels;

  int total_samples;
  int test_samples;
  int iterations=1;


  // -- Note: decode the files ...
  total_labels=ccu32_reverse(((int *)labels)[1]);

  total_images=ccu32_reverse(((int *)images)[1]);
  image_size_y=ccu32_reverse(((int *)images)[2]);
  image_size_x=ccu32_reverse(((int *)images)[3]);

  cctracelog("image size %ix%i",image_size_x,image_size_y);

  labels=(unsigned char *)labels+sizeof(int)*2;
  images=(unsigned char *)images+sizeof(int)*3;

  image_size=image_size_x*image_size_y;

  test_samples=10000;


  if(total_images!=total_labels)
  {
    cctracewar("image count %i differs from label count %i", total_images,total_labels);
  }


  total_samples=(total_images<total_labels?total_images:total_labels)-test_samples;

  vector_t target[10];
  target[0]=new_vec10(1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
  target[1]=new_vec10(0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
  target[2]=new_vec10(0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
  target[3]=new_vec10(0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
  target[4]=new_vec10(0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f);
  target[5]=new_vec10(0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f);
  target[6]=new_vec10(0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f);
  target[7]=new_vec10(0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f);
  target[8]=new_vec10(0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f);
  target[9]=new_vec10(0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f);

  network_t network;
  network_init(&network,28*28,16,10);

  float normalize=1.f/255.f;

  vector_t sample=new_vec(image_size);

  int correct_samples=0;


  // Todo: experiment with "mini-batches"
  for(int j=0;j<iterations;++j)
  { for(int i=0;i<total_samples;++i)
    { int label=labels[i];
      unsigned char *image=images+image_size*i;
      for(size_t n=0; n<sample.len; ++n)
        sample.mem[n]=normalize*image[n];
      vector_t t=target[label];
      network_train(&network,sample,t);
      int p=network_prediction(&network);

      int is_correct=label==p;
      correct_samples+=is_correct;
    }
  }

  float accuracy=100.f*(float)correct_samples/(iterations*total_samples);
  cctracelog("training accuracy: %.2f%%%%",accuracy);

  correct_samples=0;
  for(int i=total_samples;i<total_samples+test_samples;++i)
  { int label=labels[i];
    unsigned char *image=images+image_size*i;
    for(size_t n=0; n<sample.len; ++n)
      sample.mem[n]=normalize*image[n];

    network_feed(&network,sample);
    int p=network_prediction(&network);

    int is_correct=label==p;
    correct_samples+=is_correct;

    // if(!is_correct)
    // { ccprintf(CCFG_RED);
    //   ccdebuglog("Predicted %i, was %i",p,label);
    //   printimage(image_size_x,image_size_y,image);
    //   ccprintf(CCEND);
    // }
  }

  accuracy=100.f*(float)correct_samples/test_samples;
  cctracelog("test accuracy: %.2f%%%%",accuracy);

ccdbleave("main");
ccdebugend();
}
