#define KLARK_APPMODE_CONSOLE
#include "dr-include.h"

#include "dr-vec.h"
#include "num-ai.h"

static ai_vec lay_prp_fwd(ai_lay * lay, ai_vec inp)
{ mat_mul_(lay->out, lay->wei, inp     );
  vec_sub_(lay->out, lay->out, lay->bia); // TODO(RJ): there's a faster way of doing this.
  vec_sig_(lay->act, lay->out);
  // for(ai_int i = 0; i < lay->out.len; ++ i)
  // { lay->act.mem[i] = sigmoid(lay->out.mem[i]);
  // }
  return lay->act;
}

static void net_prp_fwd(
  ai_net * net,
  ai_vec   inp)
{ lay_prp_fwd(& net->lay_o,
    lay_prp_fwd(& net->lay_i, inp));
}

static void net_prp_bwd(
  ai_net * net,
  ai_vec   inp_v,
  ai_vec   tar_v)
{
  ai_lay lay_o = net->lay_o;
  ai_vec out_o = lay_o.out;
  ai_vec act_o = lay_o.act;
  ai_mat wei_o = lay_o.wei;
  ai_vec err_o = lay_o.err;
  ai_vec new_bia_o = lay_o.new_bia;
  ai_mat new_wei_o = lay_o.new_wei;
  ai_lay lay_i = net->lay_i;
  ai_vec out_i = lay_i.out;
  ai_vec act_i = lay_i.act;
  ai_mat wei_i = lay_i.wei;
  ai_vec dsg_o = new_bia_o;

  vec_sub_(err_o, lay_o.act, tar_v);
  vec_dsg_(dsg_o, dsg_o, out_o);

  for(ai_int j = 0; j < err_o.len; ++ j)
  { for(ai_int i = 0; i < out_i.len; ++ i)
    { new_wei_o.mem[j * new_wei_o.vec_max + i] = err_o.mem[j] * out_i.mem[i];
    }
  }


}

static void net_grd_dsc(
  ai_net * net,
  ai_vec   inp_v,
  ai_vec   tar_v)
{
  for(ai_int i = 0; i < net->itr; ++ i)
  { net_prp_fwd(net, inp_v);
    net_prp_bwd(net, inp_v, tar_v);
  }



}


VOID MAIN()
{ unsigned int labels_file_size;
  void *labels_file_data = LoadFileData(& labels_file_size, "data\\train-labels.idx1-ubyte");
  unsigned int label_count = int_flip(((unsigned int *)labels_file_data)[1]);
  unsigned char *label_array = (unsigned char *)labels_file_data + sizeof(int)*2;
  unsigned int images_file_size;
  void *images_file_data = LoadFileData(& images_file_size, "data\\train-images.idx3-ubyte");
  unsigned int img_num = int_flip(((unsigned int *) images_file_data)[1]);
  unsigned int img_row_len = int_flip(((unsigned int *) images_file_data)[2]);
  unsigned int img_col_len = int_flip(((unsigned int *) images_file_data)[3]);
  unsigned char * img_mem = (unsigned char *)images_file_data + sizeof(int)*3;

  { Assert(!! vec_eql(new_vec2(1, 1), new_vec2(1, 1)));
    Assert(!  vec_eql(new_vec2(1, 0), new_vec2(1, 1)));

    Assert(
      vec_eql(
        new_vec2(0.f, 0.f),
        vec_mul(
          new_vec2(0.f, 1.f),
          new_vec2(1.f, 0.f))));
    Assert(
    vec_eql(
      new_vec2(0.f, 0.f),
      vec_add(
        new_vec2(- 1.f,   1.f),
        new_vec2(  1.f, - 1.f))));
    Assert(0 == vec_dot(
      new_vec2(0.f, 1.f),
      new_vec2(1.f, 0.f)));
    Assert(0 == vec_dot(
      new_vec3(1.f, 0.f, 0.f),
      new_vec3(0.f, 1.f, 0.f)));
    Assert(0 == vec_dot(
      new_vec3(1.f, 0.f, 0.f),
      new_vec3(0.f, 0.f, 1.f)));
    Assert(0 == vec_dot(
      new_vec3(0.f, 1.f, 0.f),
      new_vec3(0.f, 0.f, 1.f)));
  }

  ai_vec valid_output_vectors[10] =
  { new_vec10(1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f), // 0
    new_vec10(0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f), // 1
    new_vec10(0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f), // 2
    new_vec10(0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f), // 3
    new_vec10(0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f), // 4
    new_vec10(0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f), // 5
    new_vec10(0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f), // 6
    new_vec10(0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f), // 7
    new_vec10(0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f), // 8
    new_vec10(0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f), // 9
  };

  ai_net net;
  net.lay_i = new_lay(28*28, 16);
  net.lay_o = new_lay(16,    10);
  net.num   = img_num;
  net.del   = 0.10f;
  net.itr   = 500;
  ai_vec   inp = new_vec(28*28);
  ai_float nor = 1.f / 255.f;
  ai_int sample_batch = 250;
  ai_int sample_count = img_num;

  // 56000 ms for all samples avg
  // 51000 ms for all samples avg (after sigmoid wide)
  // 46000 ms for all samples avg (after sigmoid prime wide)
  // for(;;)
  { TRACE_BLOCK("TIMED");
    for(ai_int sample_index = 0; sample_index < sample_count; sample_index += sample_batch)
    { for(ai_int batch_index = 0; batch_index < sample_batch; ++ batch_index)
      { ai_int image_index = sample_index + batch_index;

        const int label = label_array[image_index];
        unsigned char *img_dat = img_mem + (img_col_len * img_row_len) * image_index;

        for(size_t i = 0; i < inp.len; ++ i)
        { inp.mem[i] = nor * img_dat[i];
        }

        ai_float prd_val = -1;
        ai_int   prd_idx = -1;

        net_grd_dsc(& net, inp, valid_output_vectors[label]);

        for(ai_int i = 0; i < net.lay_o.act.len; ++ i)
        { if(prd_val < net.lay_o.act.mem[i])
          { prd_val = net.lay_o.act.mem[i];
            prd_idx = i;
          }
        }
        // TRACE_I("sample [%i:%i] prd %i, lbl %i", sample_index, batch_index, prd_idx, label);
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