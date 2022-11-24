/**
 *   Because we can't visualize fully fledged math on a primitive text editor, we're going to invent a limited and simple
 *   ASCII semi-intermediate notation that'll help me and you understand and document math related code easier and quicker.
 *   This notation is also used by a pre-processor to generate a website, you'll see the rendered math there.
 *   ---
 *   Although the math is simple, hopefully the combination of math, code and comments will help clear up any doubts you may have.
 *
 *   Let's give the notation a name, "Dayan's Notation", anything within back-ticks is Dayan's Notation.
 *
 *   Here's the quite shallow Backus–Naur form used by the parser.
 *
 * <statement-end-operator> ::= `;`
 * <unknown-operator> ::= `?`
 * <lagrange-derivative-operator> ::= `"`
 * <leibniz-derivative-operator> ::= `\\\`
 * <leibniz-partial-derivative-operator> ::= `\\`
 * <comma-operator> ::= `,`
 * <single-colon-operator> ::= `:`
 * <double-colon-operator> ::= `::`
 * <single-dot-operator> ::= `.`
 * <double-dot-operator> ::= `..`
 * <triple-dot-operator> ::= `...`
 * <string-literal> ::= `'` <string-char-sequence> `'`
 *
 * <semantic-assignment-operator> ::= <expression> <double-colon-operator> <string-literal>
 *
 * <equivalent-operator> ::= `==`
 * <equivalent-expression> ::= <expression> <equivalent-operator> <expression>
 *
 * <assignment-operator> ::= `=`
 * <assignment-expression> ::= <expression> <assignment-operator> <expression>
 *
 * <expression> <double-colon-operator> <expression>
 *
 * <group-expression> ::= `(` <expression> `)`
 *
 * <range-expression> ::= <expression> <double-dot-operator> <expression>
 *
 * <inclusive-range-expression> ::= <expression> <triple-dot-operator> <expression>
 *
 * <rank-expression> ::= `<` <expression> `>`
 *
 * <derivative-expression> ::=
 *  <expression> <leibniz-derivative-operator> <expression>
 *  <expression> <leibniz-partial-derivative-operator> <expression>
 *  <expression> <lagrange-derivative-operator> <rank-expression>(opt)
 *
 * <subscript-adjacency-list> ::=
 *  <subscript-postfix-expression>
 *  <subscript-postfix-expression> <comma-operator> <subscript-adjacency-list>
 *
 * <subscript-postfix-expression> ::=
 *  <expression>
 *  <expression> `[` <subscript-adjacency-list> `]`
 *
 * <subscript-expression> ::= <subscript-postfix-expression>
 *
 * <pow-expression> ::= `pow` `(` <function-argument-list> `)`
 * <log-expression> ::= `log` `(` <function-argument-list> `)`
 * <mul-expression> ::= `mul` `(` <function-argument-list> `)`
 * <div-expression> ::= `div` `(` <function-argument-list> `)`
 * <mod-expression> ::= `mod` `(` <function-argument-list> `)`
 * <add-expression> ::= `add` `(` <function-argument-list> `)`
 * <sub-expression> ::= `sub` `(` <function-argument-list> `)`
 * <dot-expression> ::= `dot` `(` <function-argument-list> `)`
 * <for-statement>  ::= `for` `(` <range-expression> `)` <single-colon-operator> <expression> <statement-end-operator>(opt)
 * <sum-statement>  ::= `sum` `(` <range-expression> `)` <single-colon-operator> <expression> <statement-end-operator>(opt)
 *
 *
 *   The activation function, a brief explanation:
 *   ---
 *   We use partial derivatives to find sort of a "direction", a pointer towards some minimum.
 *
 *   For a second, let's think of them as oriented pointers with some variable magnitude.
 *
 *   Initially, they are most likely pointing towards the wrong place, and so it is the job of the
 *   algorithm to ever so slightly nudge each pointer, in a direction that works not just for it,
 *   but for every other pointer it is linked to.
 *
 *   In a way, these pointers would be too credulous, too susceptible to disturbances, noise or just change
 *   in general if they were guided by the unfiltered output of a neuron.
 *
 *   By introducing that extra activation function, we're adding another step to the derivation chain and thus
 *   normalizing and filtering out noise.
 *
 *   Sigmoid also has some useful properties, for instance, values outside of the range ~ -5..5
 *   make no almost no difference, were as values between ~ -3..3 have the biggest effect.
 *
 *   Is almost like a noise filter.
 *
 *   Sigmoid is just one of the many activation functions used today, so it is in your
 *   best interest too see which one works best for you once you have a firm understanding of the concept.
 *
 *
 *   Getting started.
 *   ---
 *
 *   We're Going To Use Subscripts A Lot, Here's What They Mean.
 *
 * 0x00: d :: 'sample index'
 * 0x00: k :: 'layer index'
 * 0x00: d :: 'input index'
 * 0x00: j :: 'neuron index'
 * 0x00: i :: 'weight index'
 * 0x00: r :: 'neuron count'
 *
 *   This is the first layer, the output layer.
 *
 * 0x00: k[0]
 *
 *   This is is the last layer, the input layer. (Negative array indices).
 *
 * 0x00: k[-1]
 *
 *   The Layer After `k` (closer to the input layer)
 *
 * 0x00: `k+1 == k ++`
 *
 *   The Layer Before `k` (closer to the output layer)
 *
 * 0x00: `k-1 == k --`
 *
 *   Define our activation function to be:
 *
 * 0x00: sig(x) := 1/(1 + exp(- x))
 *
 *   And it's derivative:
 *
 * 0x00: sig"(x) := (1 - sig(x)) * sig(x)
 *
 *   Define the unfiltered output of a neuron to be:
 *
 * 0x00: A[k:j] := sum(i, W[k:j,i] * O[k+1:j,i]) + B[k:j]
 *
 *   Define the output of a neuron to be:
 *
 * 0x00: O[k:j] := sig(A[k,j])
 *
 *   Define our target output:
 *
 * 0x00: Y[d:j]
 *
 *   Define Our Cost Function, For A Single Input - Output Pair To Be:
 *
 * 0x00: E[d:j](X) := 1/2 * pow<2>(O[0:j] - Y[d:j])
 *
 *   Let's begin with the output layer: `k = 0`
 *
 *   What are we trying to do?
 *   Find how much each neuron of the output layer affects each `E[d:j]` so that
 *   we now how much to nudge each weight and bias appropriately.
 *
 *   Let's find out how much `E[d:j]` is affected by the unfiltered output `A[k:j]` of a neuron.
 *
 *   Since `E[d:j]` is a function of `O[k:j]` and not a function of `A[k:j]` directly, we have to go through `O[k:j]`,
 *   which is actually modulating `A[k:j]` through the activation function.
 *
 *   And so we get:
 *
 * 0x00: E[d:j]\\A[k:j] := E[d:j]\\O[k:j] * O[k:j]\\A[k:j]
 *
 *   The partial change in `E[d:j]` due to `A[k:j]` is the product of the change in `E[d:j]` due to `O[k:j]` and the change
 *   in `O[k:j]` due to `A[k:j]`.
 *
 *   We call this term `E[d:j]\\A[k:j]` the "error", and we'll use the dollar sign to denote it.
 *   This is what it looks fully resolved at the output layer:
 *
 * 0x00: $[k:j] := (O[k:j] - Y[k:j]) * sig"(A[k:j])
 *
 *   Since the error is defined per node at the output layer, we have to go over every one of its nodes `j`
 *   and calculate it's error.
 *
 * 0x00: for(j..r[k]):
 * 0x00:  ?.[j] = (O[k:j] - Y[k:j]) * sig"(A[k:j])
 *
 *   Now that we have the `$` term, we know how much the cost function changes due to `A[k:j]`.
 *   But we need to go further, how much does `A[k:j]` change due to `B[k:j]` and every single `W[k:j,i]`.
 *
 *   Change In `A[k:j]` due to `B[k:j]`.
 *
 *   For this one we don't have to do much.
 *
 * 0x00: A[k:j]\\B[k:j] := $[k:j] * 1 == $[k:j];
 *
 *
 *   Change In `A[k:j]` due to `W[k:j,i]`
 *
 *   This one is a little bit more involved.
 *
 * 0x00: A[k:j]\\W[k:j,i] := $[k:j] * O[k+1:i]
 *
 *   Let's expand this expression a bit, to see how it works.
 *
 * 0x00: A[k:0]\\W[k:  0, 0  ] := $[k:0] * O[k+1:0]
 * 0x00: A[k:0]\\W[k:  0, 1  ] := $[k:0] * O[k+1:1]
 * 0x00: A[k:0]\\W[k:  0, 2  ] := $[k:0] * O[k+1:2]
 * 0x00: A[k:0]\\W[k:  0, 3  ] := $[k:0] * O[k+1:3]
 * 0x00: A[k:0]\\W[k:  0, .. ] := $[k:0] * O[k+1:3]
 * 0x00: A[k:0]\\W[k:  1, .. ] := ..
 * 0x00: A[k:0]\\W[k:  2, .. ] := ..
 * 0x00: A[k:0]\\W[k:  3, .. ] := ..
 * 0x00: A[k:0]\\W[k: .., .. ] := ..
 *
 *   We see that `i` corresponds to the weight `i` of node `j` of layer `k` AND to the output of node `i` in layer `k+1`.
 *   Each weight is responsible for one of the outputs of the next layer.
 *
 *   And so the change in `A[k=0:j=0]` due to `W[k=0:j=0,i]`, is how much `O[k+1:i]` is modulating that weight.
 *
 *   Here's the psuedo code that computes `A[0:j]\\W[0:j,i]`.
 *
 * 0x00: for(j..r[k]):
 * 0x00:  for(i..r[k+1]):
 * 0x00:   m[j][i] = $[k:j] * O[k+1:i]
 *
 *   Now, why is `O[k+1:i]` the derivative of `A[k:j]\\W[k:j,i]`?
 *
 *   A node, is a first-degree polynomial, exactly like this one:
 *
 * 0x00: A[k:j] =
 * 0x00:   W[k:0,i=0] * O[k+1:i=0] +
 * 0x00:   W[k:0,i=1] * O[k+1:i=1] +
 * 0x00:   W[k:0,i=2] * O[k+1:i=2] + ...;
 *
 *   The only difference is, that we use the `sum()` operator instead.
 *
 * 0x00: A[k:0]\\W[k:0,i=0] := O[k+1:i=0]
 * 0x00: A[k:0]\\W[k:0,i=1] := O[k+1:i=1]
 * 0x00: A[k:0]\\W[k:0,i=2] := O[k+1:i=2]
 *
 *
 *   Hidden Layers, `k > 0`:
 *   ---
 *
 *   First let's once again establish that what we're ultimatly trying to do:
 *   Find out how much our weights and biases have affected `E[d:j]`.
 *
 *   This might sound as having to compute a long derivation chain that extends all the way back to the final layer,
 *   but it doesn't have to be that way.
 *
 *   Let's instead accumulate the `$` term, as we move away from the output layer. As if we were instead bring the
 *   error with us. We only focus on how much we've affected the previous outputs.
 *
 *   Now, every single weight, of every single neuron, affects every single output of the previous layer.
 *   So if we're ultimatly trying to compute `E[d:l]\\A[k:j]`, we have to also compute how much `A[k:j]` has affected
 *   every `A[k-1:j=0..r[k-1]]`.
 *
 *
 *   E[d:l]\\A[k:j] := sum(l..r[k-1], $[k-1:l] * A[k-1:l]\\A[k:l])
 *
 *   $[k:j] := sum(l..r[k-1], $[k-1:l] * A[k-1:l]\\A[k:l])
 *
 *
 *
 *   Pseudo Code Version
 *
 * 0x00:
 *  for(j..r[k]):
 *   .[i] = sum(l..r[k-1]): $[k-1:l] * W[k-1:l:j] * sig"(a[k:j])
 *
 * ### C Version
 * ### Change In E Due To W
 * 0x00: E\\W[k:j:i] = $[k:j] * O[k+1:i]
 * 0x00:
 *  for(j..r[k]):
 *    for(i..r[k+1]):
 *     .[j][i] = $[k:j] * O[k+1:i]
 * ### C Version
 **/
#define KLARK_APPMODE_CONSOLE
#include "dr-include.h"

#include "dr-vec.h"
#include "num-ai.h"


static ai_vec __forceinline __vectorcall lay_prp_fwd(ai_lay * lay, ai_vec inp)
{ mat_dot_(lay->out, lay->wei, inp     );
  vec_add_(lay->out, lay->out, lay->bia);
  vec_sig_(lay->act, lay->out);
  return lay->act;
}

static void lay_upd(ai_lay *lay, ai_float alpha)
{ // TODO(RJ): SPEED!
  ai_vec bia_old = lay->bia;
  ai_vec bia_new = lay->new_bia;
  for(ai_int x = 0; x < bia_old.len; ++ x)
  { ai_float y = -alpha * bia_new.mem[x];
    bia_old.mem[x] += y;
  }
  ai_mat wei_old = lay->wei;
  ai_mat wei_new = lay->new_wei;
  for(ai_int x = 0; x < wei_old.min * wei_old.vec_max; ++ x)
  { ai_float y = -alpha * wei_new.mem[x];
    wei_old.mem[x] += y;
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

  vec_sub_(err_o, act_o, tar_v);
  vec_dsg_(tmp_o, out_o);
  vec_mul_(err_o, err_o, tmp_o);
  for(ai_int j = 0; j < wei_o.row; ++ j)
  { for(ai_int i = 0; i < wei_o.col; ++ i)
    { new_wei_o.mem[j * new_wei_o.vec_max + i] = err_o.mem[j] * act_i.mem[i];
    }
  }

  vec_dsg_(err_i, out_i);
  for(ai_int j = 0; j < wei_i.row; ++ j)
  { ai_float acc = 0;
    for(ai_int l = 0; l < wei_o.row; ++ l)
    { acc += err_o.mem[l] * wei_o.mem[l * wei_o.vec_max + j];
    }
    err_i.mem[j] *= acc;
  }
  for(ai_int j = 0; j < wei_i.row; ++ j)
  { for(ai_int i = 0; i < wei_i.col; ++ i)
    { new_wei_i.mem[j * new_wei_i.vec_max + i] = err_i.mem[j] * inp_v.mem[i];
    }
  }
  // Sleep(256);
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


    ai_mat mat = new_row_mat(2,2);
    mat[0][0] = 1.f; mat[0][1] = 2.f;
    mat[1][0] = 3.f; mat[1][1] = 4.f;

    Assert(
      vec_eql(
        new_vec2(3.f,7.f),
        mat_dot(mat, new_vec2(1.f,1.f))));

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


  for(;;)
  { TRACE_BLOCK("TIMED");

    for(ai_int sample_index = 0; sample_index < sample_count; sample_index += sample_batch)
    {
      ai_float correct_predictions = 0.f;
      for(ai_int batch_index = 0; batch_index < sample_batch; ++ batch_index)
      { ai_int image_index = sample_index + batch_index;

        const int label = label_array[image_index];

        // if(label != 0)
        // { continue;
        // }

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

        if(label == prd_idx)
        { correct_predictions ++;
        } else
        { // correct_predictions --;
        }


        // TRACE_I("sample [%i:%-4i] lbl %i ->> prd %i %s", sample_index, batch_index, accuracy, label, prd_idx,
        //   (prd_idx==label) ? L"SUCCESS" : L"FAILED");

        lay_upd(& net.lay_i, .1f);
        lay_upd(& net.lay_o, .1f);

        // Sleep(10);

      }
      ai_float accuracy = 100.f * (correct_predictions / sample_batch);
      TRACE_I("accuracy %%%-3.2f", accuracy);
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