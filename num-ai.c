
#define KLARK_APPMODE_CONSOLE
#include "dr-include.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static unsigned int __forceinline int_flip(unsigned i)
{
  return (((i & 0xFF000000) >> 0x18) |
          ((i & 0x000000FF) << 0x18) |
          ((i & 0x00FF0000) >> 0x08) |
          ((i & 0x0000FF00) << 0x08));
}


static float input_layer[28*28/*748*/];
static float activation_layer1[16];
static float activation_layer1_biases[16];
static float activation_layer1_weights[16][28*28/*748*/];
static float activation_layer3[10];
static float activation_layer3_biases[10];
static float activation_layer3_weights[10][16];



static void forward_propagation(unsigned char *pixel_data)
{
  for(size_t i = 0; i < 28*28; ++ i)
  { input_layer[i] = pixel_data[i] / 255.f;
  }

  for(size_t i = 0; i < 16; ++ i)
  { size_t acc = 0;
    for(size_t j = 0; j < 28*28; ++ j)
    { acc += input_layer[i] * activation_layer1_weights[i][j];
     }
    acc -= activation_layer1_biases[i];
    activation_layer1[i] = acc < 0 ? 0 : acc;
  }
}


static double rand_f(double min, double max)
{ return min + ((double)rand() / (double)RAND_MAX) * (max - min);
}


VOID MAIN()
{
  unsigned int labels_file_size;
  void *labels_file_data = LoadFileData(& labels_file_size, "data\\train-labels.idx1-ubyte");
  unsigned int label_count = int_flip(((unsigned int *)labels_file_data)[1]);
  unsigned char *label_array = (unsigned char *)labels_file_data + sizeof(int)*2;

  unsigned int images_file_size;
  void *images_file_data = LoadFileData(& images_file_size, "data\\train-images.idx3-ubyte");
  unsigned int image_num = int_flip(((unsigned int *) images_file_data)[1]);
  unsigned int image_row_len = int_flip(((unsigned int *) images_file_data)[2]);
  unsigned int image_col_len = int_flip(((unsigned int *) images_file_data)[3]);
  unsigned char * image_mem = (unsigned char *)images_file_data + sizeof(int)*3;





  // float activation_layer2[16];
  // float activation_layer2_biases[16];
  // float activation_layer2_weights[16 * 16];

  for(int i = 0; i < 16; ++ i)
  { activation_layer1_biases[i] = rand_f(-.5f, .5f);
  }
  for(int i = 0; i < 16; ++ i)
  { for(size_t j = 0; j < 28*28; ++ j)
    { activation_layer1_weights[i][j] = rand_f(-.5f, .5f);
    }
  }

  for(int i = 0; i < 10; ++ i)
  { activation_layer3_biases[i] = rand_f(-.5f, .5f);
  }
  for(int i = 0; i < 10; ++ i)
  { for(size_t j = 0; j < 16; ++ j)
    { activation_layer3_weights[i][j] = rand_f(-.5f, .5f);
    }
  }

  for(int i = 0; i < image_num && i < 16; ++ i)
  { unsigned char *pixel_data = image_mem + i * (image_col_len * image_row_len);




    // for(size_t i = 0; i < 10; ++ i)
    // { size_t acc = 0;
    //   for(size_t j = 0; j < 16; ++ j)
    //   { acc += activation_layer1[i] * activation_layer2_weights[i][j];
    //   }
    //   acc -= activation_layer3_biases[i];
    //   activation_layer1[i] = acc < 0 ? 0 : acc;
    // }




    // if(! stbi_write_png(FormatA("data\\sample\\image_%i.png", i), image_col_len, image_row_len, 1, pixel_data, 28))
    // { TRACE_E("failed to write image");
    //   __debugbreak();
    // }
  }

  UnloadFileData(labels_file_data);
  UnloadFileData(images_file_data);


  __debugbreak();

}