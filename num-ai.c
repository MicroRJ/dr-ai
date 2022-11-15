#include "num-ai.h"

static void net_fprop(
  ai_net * net,
  ai_vec   inp_v)
{
  mat_mul_v(& net->lay_i,   inp_v     );
  mat_mul_m(& net->lay_o, & net->lay_i);
}

static void net_bprop(
  ai_net * net,
  ai_vec   inp_v,
  ai_vec   tar_v)
{
  mat_mul_v(& net->lay_i,   inp_v     );
  mat_mul_m(& net->lay_o, & net->lay_i);
}

static void net_dscnt(
  ai_net * net,
  ai_vec   inp_v,
  ai_vec   tar_v)
{
  mat_rnd(& net->lay_i);
  mat_rnd(& net->lay_o);

  for(ai_int i = 0; i < net->itr; ++ i)
  { net_fprop(net, inp_v);
    net_bprop(net, inp_v, tar_v);
  }
}


VOID MAIN()
{
  // These vectors just match to a tag.
  // ML ppl call these one-hot matrices, or vectors in this case, since only one bit or value is set to one.
  ai_float valid_result_data[10][16] = // allocate at least 16 bytes to help with wide instructions.
  { {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, // 0
    {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, // 1
    {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, // 2
    {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, // 3
    {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0}, // 4
    {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0}, // 5
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0}, // 6
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0}, // 7
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0}, // 8
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}, // 9
  };
  ai_vec valid_output_vectors[10] =
  { {10, valid_result_data[0]}, {10, valid_result_data[1]},
    {10, valid_result_data[2]}, {10, valid_result_data[3]},
    {10, valid_result_data[4]}, {10, valid_result_data[5]},
    {10, valid_result_data[6]}, {10, valid_result_data[7]},
    {10, valid_result_data[8]}, {10, valid_result_data[9]},
  };

  unsigned int labels_file_size;
  void *labels_file_data = LoadFileData(& labels_file_size, "data\\train-labels.idx1-ubyte");
  unsigned int label_count = int_flip(((unsigned int *)labels_file_data)[1]);
  unsigned char *label_array = (unsigned char *)labels_file_data + sizeof(int)*2;
  unsigned int images_file_size;
  void *images_file_data = LoadFileData(& images_file_size, "data\\train-images.idx3-ubyte");
  unsigned int img_num = int_flip(((unsigned int *) images_file_data)[1]);
  unsigned int img_row_len = int_flip(((unsigned int *) images_file_data)[2]);
  unsigned int img_col_len = int_flip(((unsigned int *) images_file_data)[3]);
  unsigned char * img_mem = (unsigned char *)images_file_data + sizeof(int)*3;

  ai_net net;
  net.lay_i = mat_new(28*28, 16);
  net.lay_o = mat_new(16,    10);
  net.num   = img_num;
  net.dpi   = 0.10f;
  net.itr   = 500;
  net.reg   = vec_new(28*28 + 16*4);

  ai_vec   inp = vec_new(28*28);
  ai_float nor = 1.f / 255.f;

  { TRACE_BLOCK("TIMED");

    for(int img_index = 0; img_index < 15; /*img_num - 10000*/ ++ img_index)
    { const int label = label_array[img_index];
      unsigned char *img_dat = img_mem + (img_col_len * img_row_len) * img_index;
      for(size_t i = 0; i < inp.len; ++ i)
      { inp.mem[i] = nor * img_dat[i];
      }

      // if(! stbi_write_png(FormatA("data\\sample\\image_%i.png", i), image_col_len, image_row_len, 1, pixel_data, 28))
      // { TRACE_E("failed to write image");
      //   __debugbreak();
      // }
    }
  }


  UnloadFileData(labels_file_data);
  UnloadFileData(images_file_data);
  __debugbreak();
}