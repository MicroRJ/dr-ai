/**
 *
 * Because we can't visualize fully fledged math on a primitive text editor,
 * we're going to invent a limited and simple ASCII semi-intermediate notation
 * that'll help me and you understand and document math related code easier and quicker.
 * This notation is also used by a pre-processor to generate the website, you'll see the rendered math there.
 *
 * Although the math is simple, hopefully the combination of math, code and comments will help
 * clear up any doubts you may have.
 *
 * Let's give the notation a name, "Dayan's Notation", anything within back-ticks is Dayan's Notation.
 *
 * Here's the quite shallow Backus–Naur form.
 * <statement-end-operator> ::= `;`
 * <lagrange-derivative-operator> ::= `"`
 * <leibniz-derivative-operator> ::= `\\\`
 * <leibniz-partial-derivative-operator> ::= `\\`
 * <comma-operator> ::= `,`
 * <colon-operator> ::= `:`
 * <single-dot-operator> ::= `.`
 * <double-dot-operator> ::= `..`
 * <triple-dot-operator> ::= `...`
 * <range-expression> ::= <expression> <double-dot-operator> <expression>
 * <inclusive-range-expression> ::= <expression> <triple-dot-operator> <expression>
 * <rank-expression> ::= `<` <expression> `>`
 * <derivative-expression> ::=
 *  <expression> <leibniz-derivative-operator> <expression>
 *  <expression> <leibniz-partial-derivative-operator> <expression>
 *  <expression> <lagrange-derivative-operator> <rank-expression>(opt)
 * <subscript-argument> ::=
 *  <expression>
 *  <expression> <comma-operator> <subscript-argument>
 *  <subscript-argument>(opt) <colon-operator> <subscript-argument>(opt)
 * <subscript-expression> ::= `[` <subscript-argument>(opt) `]`
 * <for-statement> ::= `for` `(` <range-expression> `)` <colon-operator> <expression> <statement-end-operator>(opt)
 * <sum-statement> ::= `sum` `(` <range-expression> `)` <colon-operator> <expression> <statement-end-operator>(opt)
 **/
#define KLARK_APPMODE_CONSOLE
#include "dr-include.h"

#include "dr-vec.h"
#include "num-ai.h"


static ai_vec __forceinline __vectorcall lay_prp_fwd(ai_lay * lay, ai_vec inp)
{ mat_dot_(lay->out, lay->wei, inp     );
  vec_sub_(lay->out, lay->out, lay->bia);
  vec_sig_(lay->act, lay->out);
  return lay->act;
}

static void lay_upd(ai_lay *lay, ai_float alpha)
{ // TODO(RJ): SPEED!
  ai_vec bia_old = lay->bia;
  ai_vec bia_new = lay->new_bia;
  for(ai_int x = 0; x < bia_old.len; ++ x)
  { bia_old.mem[x] += -alpha * bia_new.mem[x];
  }
  ai_mat wei_old = lay->wei;
  ai_mat wei_new = lay->new_wei;
  for(ai_int x = 0; x < wei_old.min * wei_old.vec_max; ++ x)
  { wei_old.mem[x] += -alpha * wei_new.mem[x];
  }
}

static void __forceinline net_prp_fwd(ai_net *net, ai_vec inp)
{ lay_prp_fwd(& net->lay_o,
    lay_prp_fwd(& net->lay_i, inp));
}

static void __vectorcall net_prp_bwd(ai_net * net, ai_vec inp_v, ai_vec tar_v)
{ ai_lay
    lay_o = net->lay_o,
    lay_i = net->lay_i;
  ai_vec
    out_o = lay_o.out,
    act_o = lay_o.act,
    err_o = lay_o.err,
    tmp_o = lay_o.tmp,

    out_i = lay_i.out,
    act_i = lay_i.act,
    err_i = lay_i.err,
    tmp_i = lay_i.tmp;
  ai_mat
    wei_o     = lay_o.wei,
    wei_i     = lay_i.wei,
    new_wei_o = lay_o.new_wei,
    new_wei_i = lay_i.new_wei;
  //
  // ### Define our semantic indices.
  // 0x00: `k := "layer"       `
  // 0x00: `j := "neuron"      `
  // 0x00: `i := "index"       `
  // 0x00: `r := "neuron count"`
  // ### This is the first layer, the output layer.
  // 0x00: `k[  0]`
  // ### This is is the last layer, the input layer.
  // 0x00: `k[- 1]`
  // ### The Layer After `k` (closer to the input layer)
  // 0x00: `k+1 == k ++ `
  // ### The Layer Before `k` (closer to the output layer)
  // 0x00: `k-1 == k -- `
  // ### Define Our Cost Function, For A Single Input - Output Pair To Be:
  // 0x00: E[k:j] := 1/2 * (A[k:j] - Y[k:j])^2
  // ### Define Our Activation Function To Be:
  // 0x00: sig(x) := 1/(1 + exp(- x))
  // ### Define The Derivative Of That Function To Be:
  // 0x00: sig"(x) := (1 - exp(x)) * sig(x)
  // ### Define The Output Of A Neuron To Be:
  // 0x00: A[k:j] := sum(i, W[k:j-i] * [k+1:j-i]) + B[k:j]
  // ### Define The Output Of A Neuron, After Passed Through Activation Function To Be:
  // 0x00: O[k:j] := sig(A)
  // ### Define the "error".
  // 0x00: $[k:j] := E\\A[k:j] = (A[k:j] - Y[k:j]) * sig"(A[k:j])
  // ### Pseudo Code Version
  // 0x00: for(j..r[k]):
  //        .[j] = sig"(A[k:j]) * (O[k:j] - Y[k:j])
  // ### Change In `A[k:j]` due to `B[k:j]`
  // 0x00: A[k:j]\\B[k:j] := $[k:j] * 1
  // ### Change In `A[k:j]` due to `W[k:j-i]`
  // 0x00: A[k:j]\\W[k:j-i] := $[k:j] * O[k+1:i]
  // ### Pseudo Code Version
  // 0x00: for(j..r[k]):
  //        for(i..r[k+1]):
  //          .[j][i] = $[k:j] * O[k+1:i]
  {
    // 0x00:
    //   Calculate error `$[0:0..r[0]]` of output layer.
    //   We have multiple output neurons, so we calculate the error for each.
    //   To do this we use vector operations.
    vec_sub_(err_o, act_o, tar_v);
    vec_dsg_(tmp_o, out_o);
    vec_mul_(err_o, err_o, tmp_o);
    for(ai_int j = 0; j < wei_o.row; ++ j)
    { for(ai_int i = 0; i < wei_o.col; ++ i)
      { new_wei_o.mem[j * new_wei_o.vec_max + i] = err_o.mem[j] * act_i.mem[i];
      }
    }
  }
  { // Now we're not at the output layer anymore, thus `k > 0`.
    // ### Change In E Due To A (This term is commonly known as the "error"):
    // 0x00: $[k:j] := sum(l, $[k-1:l] * W[k-1:l:j] * sig"(a[k:j]))
    // ### Pseudo Code Version
    // 0x00:
    //  for(i..r[k]):
    //   .[i] = sig"(a[k:j]) * sum(l in r[k-1]): $[k-1:j] * W[k-1:l:i]))
    // ### C Version
    vec_dsg_(err_i, out_i);
    for(ai_int i = 0; i < wei_i.row; ++ i)
    { ai_float acc = 0;
      for(ai_int l = 0; l < wei_o.row; ++ l)
      { acc += err_o.mem[l] * wei_o.mem[l * wei_o.vec_max + i];
      }
      err_i.mem[i] *= acc;
    }
    // ### Change In E Due To W
    // 0x00: E\\W[k:j:i] = $[k:j] * O[k+1:i]
    // 0x00:
    //  for(j..r[k]):
    //    for(i..r[k+1]):
    //     .[j][i] = $[k:j] * O[k+1:i]
    // ### C Version
    for(ai_int j = 0; j < wei_i.row; ++ j)
    { for(ai_int i = 0; i < wei_i.col; ++ i)
      { new_wei_i.mem[j * new_wei_i.vec_max + i] = err_i.mem[j] * inp_v.mem[i];
      }
    }
  }

  Sleep(256);
}

static void net_grd_dsc(
  ai_net * net,
  ai_vec   inp_v,
  ai_vec   tar_v)
{
  net_prp_fwd(net, inp_v);
  net_prp_bwd(net, inp_v, tar_v);
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

        memset(net.lay_i.new_bia.mem, 0,
          sizeof(ai_float) * net.lay_i.new_bia.len);
        memset(net.lay_i.new_wei.mem, 0,
          sizeof(ai_float) * net.lay_i.new_wei.min * net.lay_i.new_wei.vec_max);

        memset(net.lay_o.new_bia.mem, 0,
          sizeof(ai_float) * net.lay_o.new_bia.len);
        memset(net.lay_o.new_wei.mem, 0,
          sizeof(ai_float) * net.lay_o.new_wei.min * net.lay_o.new_wei.vec_max);

        net_grd_dsc(& net, inp, valid_output_vectors[label]);

        ai_float prd_val = -1;
        ai_int   prd_idx = -1;
        for(ai_int i = 0; i < net.lay_o.act.len; ++ i)
        { if(net.lay_o.act.mem[i] > prd_val)
          { prd_val = net.lay_o.act.mem[i];
            prd_idx = i;
          }
        }

        lay_upd(& net.lay_i, .01f);
        lay_upd(& net.lay_o, .01f);

        TRACE_I("sample [%i:%i] lbl %i ->> prd %i", sample_index, batch_index, label, prd_idx);
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


// * Example A: `W[k:j,i]`
//    Visually, `k` would be above `W` and `j` below, `i` would be next to `j`.
// * Example B: `W[k[c:v]:j,i]`
//    Visually, `k` would be above `W` and `j` below, `i` would be next to `j`,
//    `c` would be above `k` and `v` below.
//   Subscript Inference:
//   Only when repetitiveness becomes obfuscating, AND if by pattern of repetition, the context
//   is unambiguous, AND it is in the best interest of the reader, it is then allowed to omit
//   the <subscript-argument> or one if its <operands>.
//   The subscript however, must be representative of the number of expressions there are in it.
// * Example C & D:
//     `w[k:j:0]` = `w[::0]`
//     `w[k]`     = `w[]`
// Let's rewrite this expression instead as:
//
// 0x00: $[k:j] := sig"(a[k:j]) * sum(l, $[k-1:l] * W[k-1:l:j])
// Let's expand this expression to see how it works:
//
// 0x00:
//  $[k:0] := sig"(a[k:0]) *
//    ( $[k-1: 0]* W[k-1 : 0: 0] +
//      $[   : 1]* W[    :  : 0] +
//      $[   : 2]* W[    :  : 0] +
//      $[   : 3]* W[    :  : 0] )
//
// Notice how the weight index is the same as j, 0. That is
// because we're only concerned with the weight that is scaling our output.
// All the other variables can be treated as 0's. {{explanation}}
//
//
// If you're somehow thinking about a matrix transposition, you're in the right path.
//   0  1  2  3  4     0  1  2
// 0 [W][W][W][W][W]   [W][W][W] 0 <
// 1 [W][W][W][W][W] T [W][W][W] 1 <
// 2 [W][W][W][W][W]   [W][W][W] 2 <
//    ^  ^  ^  ^  ^    [W][W][W] 3 <
//                     [W][W][W] 4 <
//
//
// To transpose a matrix is to rotate it. This could have a deeper meaning, depending on how you
// interpret them, e.i Linear Algebra, but to us, it means nothing but facilitating an operation.
// So no, it has nothing to do with going "backwards", let's stop lollygagging on 3Blue1Brown
// analogies and come back to the harsh reality, it's just a coincidence, believe me, I am not a mathematician.
//
// For the 'fast' implementation however, we're not going to do this, because transposing ~4KB worth of matrix
// every iteration sounds almost slower than Visual Studio.
// (28*28*4 + 16*10*4 + 10*10*4)/1024
// The whole point of transposing a matrix is to take advantage of SIMD, making memory that was sparse, contiguous.
// And you could make the case that transposing a matrix is a good way of doing this, and it is, if you don't know
// what you're talking about.
//
// Now, we already have SIMD working for the forward propagation stage, just not for the backwards one.
// So it's clear that we'll either have to make some sort of compromise. Let's not dwell on this too
// much though, let's write a loop instead and we'll figure something out later.