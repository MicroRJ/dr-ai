#ifndef NUM_AI_HEADER_FILE
#define NUM_AI_HEADER_FILE


#define KLARK_APPMODE_CONSOLE
#include "dr-include.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// TODO(RJ):
// ; Regenerate the file data to be little endian, use 2 bits for the image data instead of 8.
// ; Compress the images using RLE base-64 or something and put them a header file that can be distributed with the source code.
// ; Better random number generator.
// ; Remove ai_lane_collapse_additive
/**
 * Resources used:
 * https://www.youtube.com/watch?v=aircAruvnKk
 * https://www.youtube.com/watch?v=IHZwWFHWa-w
 * https://www.youtube.com/watch?v=Ilg3gGewQ5U
 * https://www.youtube.com/watch?v=tIeHLnjs5U8
 * https://www.khanacademy.org/math/ap-calculus-ab/ab-limits-new/ab-1-2/v/introduction-to-limits-hd
 * https://www.khanacademy.org/math/differential-calculus/dc-diff-intro/dc-diff-calc-intro/v/newton-leibniz-and-usain-bolt?modal=1
 * https://www.khanacademy.org/math/differential-calculus/dc-diff-intro/dc-derivative-intro/v/calculus-derivatives-1-new-hd-version
 * https://www.khanacademy.org/math/differential-calculus/dc-diff-intro/dc-combine-power-rule-with-others/v/differentiating-polynomials-example?modal=1
 **/

#define SSE_SUPPORT
#define AVX_256_SUPPORT

typedef float ai_float;
typedef int ai_int;

#if defined(__cplusplus)
# define ai_vec_func(return_type) static return_type __forceinline __vectorcall
#else
# define ai_vec_func(return_type) static return_type __forceinline __vectorcall
#endif

#if defined(AVX_256_SUPPORT)
# define ai_lane          __m256
# define ai_lane_load     _mm256_load_ps
# define ai_lane_mul      _mm256_mul_ps
# define ai_lane_add      _mm256_add_ps
# define ai_lane_mul_add  _mm256_fmadd_ps
#define ai_lane_collapse_additive(l)\
  (l.m256_f32[0x00] + l.m256_f32[0x01] + l.m256_f32[0x02] + l.m256_f32[0x03] +\
   l.m256_f32[0x04] + l.m256_f32[0x05] + l.m256_f32[0x06] + l.m256_f32[0x07])
#elif defined(SSE_SUPPORT)
# define ai_lane         __m128
# define ai_lane_load    _mm_load_ps
# define ai_lane_mul     _mm_mul_ps
# define ai_lane_add     _mm_add_ps
# define ai_lane_mul_add _mm_fmadd_ps
#define ai_lane_collapse_additive(l)\
  (l.m128_f32[0x00] + l.m128_f32[0x01] + l.m128_f32[0x02] + l.m128_f32[0x03])
#endif

#define PI_APPROX (22.0/7.0)

#ifndef arr_len
# define arr_len(arr) ((int)((arr) ? (sizeof(arr)/sizeof((arr)[0])) : -1))
#endif
#ifndef sigmoid_approx
# define sigmoid_approx(o) (1.0/1.0+pow(PI_APPROX,o))
#endif
#ifndef sigmoid
# define sigmoid(o) (1.0/(1.0+exp(-o)))
#endif
#ifndef sigmoid_prime
# define sigmoid_prime(o) (sigmoid(o)*(1.0-exp(o)))
#endif
#ifndef activation_function
# define activation_function sigmoid
#endif
#ifndef activation_function
# define activation_function_prime sigmoid_prime
#endif
// TODO(RJ)!
#ifndef rand_f
# define rand_f(min,max) ((min)+((max)-(min)) * (rand()/(ai_float)RAND_MAX))
#endif

ai_vec_func(ai_float) squaref(ai_float val) { return val*val; }
ai_vec_func(ai_float) cube(ai_float val) { return val*val*val; }

// NOTE(RJ): For speed and performance, allocate vectors of at least 16 floats. This is to be compatible to AVX512
// intrinsic instructions.
typedef struct ai_vec
{ ai_int     len;
  ai_float * mem;
} ai_vec;

typedef struct ai_mat
{ // * neuron/row 0x00: [act][out][bia][wei0][wei1][....]
  // * neuron/row 0x01: [act][out][bia][wei0][wei1][....]
  // * neuron/row 0x02: [act][out][bia][wei0][wei1][....]
  // * neuron/row 0x03: [act][out][bia][wei0][wei1][....]
  // * neuron/row 0x04: [act][out][bia][wei0][wei1][....]
  // * neuron/row 0x05: [act][out][bia][wei0][wei1][....]
  // * neuron/row 0x06: [act][out][bia][wei0][wei1][....]
  // * neuron/row ....: [act][out][bia][wei0][wei1][....]
  ai_int col_len;
  ai_int row_len;
  ai_vec vec,
    act, out,
    bia, wei;
} ai_mat;

typedef struct ai_net
{ ai_mat   lay_i;
  ai_mat   lay_o;
  ai_int   num;
  ai_int   itr;
  ai_float dpi;
  ai_vec   reg;
} ai_net;

static void __forceinline vec_free(ai_vec vec)
{
  _aligned_free(vec.mem);
}

static ai_int __forceinline vec_sze_for(int len)
{ return (len + 0x0F) / 0x10 * 0x10;
}

static ai_vec __forceinline vec_new(int len)
{ ai_vec v;
  v.len = len;
  v.mem = (ai_float *) _aligned_malloc(sizeof(ai_float) * vec_sze_for(len), 0x20);
  memset(v.mem, 0, sizeof(ai_float) * len);
  return v;
}

static ai_vec vec_cln(ai_vec v)
{
  ai_vec c = vec_new(v.len);
  memcpy(c.mem, v.mem, sizeof(v.mem) * v.len);
  return c;
}

static ai_mat mat_new(int col_len, int row_len)
{ ai_mat m;
  m.col_len = col_len;
  m.row_len = row_len;
  m.act.len = row_len;
  m.out.len = row_len;
  m.bia.len = row_len;
  m.wei.len = row_len*col_len;
  m.vec = vec_new(m.act.len + m.out.len + m.bia.len + m.wei.len);
  m.act.mem = m.vec.mem + row_len*0;
  m.out.mem = m.vec.mem + row_len*1;
  m.bia.mem = m.vec.mem + row_len*2;
  m.wei.mem = m.vec.mem + row_len*3;
  return m;
}

// Extract a row from the weights.
ai_vec_func(ai_vec) mat_row(ai_mat *mat, int row)
{ ai_vec v;
  v.mem = mat->wei.mem + mat->col_len * row;
  v.len = mat->col_len;
  return v;
}

ai_vec_func(void) vec_sub(ai_vec *dst, ai_vec lhs, ai_vec rhs)
{ for(int i = 0; i < dst->len; ++ i)
  { dst->mem[i] = lhs.mem[i] - rhs.mem[i];
  }
}

ai_vec_func(ai_float) vec_dot(ai_vec lhs, ai_vec rhs)
{
  Assert(lhs.len == rhs.len);
#if defined(ai_lane)
  const size_t lane_size_in_bytes = sizeof(ai_lane)/sizeof(ai_float);
  if((lhs.len % lane_size_in_bytes) != 0) // TODO(RJ): make this proper!
  { ai_float result = 0;
    for(int i = 0; i < lhs.len; ++ i)
    { result += lhs.mem[i]*rhs.mem[i];
    }
    return result;
  }
  ai_lane lane_acc = ai_lane_add(ai_lane_load(lhs.mem),ai_lane_load(rhs.mem));
  for(int i = lane_size_in_bytes; i < lhs.len; i += lane_size_in_bytes)
  { lane_acc = ai_lane_mul_add(ai_lane_load(lhs.mem + i),ai_lane_load(rhs.mem + i),lane_acc);
  }
  // TODO(RJ)!
  return ai_lane_collapse_additive(lane_acc);
#else
  ai_float result = 0;
  for(int i = 0; i < lhs.len; ++ i)
  { result += lhs.mem[i]*rhs.mem[i];
  }
  return result;
#endif
}

ai_vec_func(ai_vec) mat_mul_v(ai_mat *dst, ai_vec src)
{ for(int row = 0; row < dst->row_len; ++ row)
  { ai_float out = vec_dot(mat_row(dst,row),src) + dst->bia.mem[row];
    ai_float act = activation_function(out);
    dst->out.mem[row] = out;
    dst->act.mem[row] = act;
  }
  return dst->out;
}

ai_vec_func(ai_vec) mat_mul_m(ai_mat *dst, ai_mat *src)
{ for(int row = 0; row < dst->row_len; ++ row)
  { mat_mul_v(dst, src->out);
  }
  return src->out;
}

ai_vec_func(void) vec_rnd(ai_vec dst)
{ for(int i = 0; i < dst.len; ++ i)
    dst.mem[i] = rand_f(0.f, 1.f);
}

ai_vec_func(void) mat_rnd(ai_mat *dst)
{ // vec_rnd(dst->bia);
  for(int i = 0; i < dst->row_len; ++ i)
    vec_rnd(mat_row(dst, i));
}

ai_vec_func(ai_float) mat_cst(ai_mat *dst, ai_vec cmp)
{ ai_float cst_acc = 0;
  for(int row = 0; row < dst->row_len; ++ row)
  { cst_acc += squaref(dst->out.mem[row] - cmp.mem[row]);
  }
  return cst_acc;
}

static int mat_prd(ai_mat *dst)
{ ai_float best_prediction_score = - 0x01;
  int best_prediction_index = - 0x01;
  for(int row_index = 0x00; row_index < dst->row_len; ++ row_index)
  { ai_float activation_score = dst->out.mem[row_index];
    if(activation_score > best_prediction_score)
    { best_prediction_score = activation_score;
      best_prediction_index = row_index;
    }
  }
  return best_prediction_index;
}


static unsigned int __forceinline int_flip(unsigned i)
{ return (((i & 0xFF000000) >> 0x18) |
          ((i & 0x000000FF) << 0x18) |
          ((i & 0x00FF0000) >> 0x08) |
          ((i & 0x0000FF00) << 0x08));
}


static ai_float nand(ai_float lhs, ai_float rhs)
{ ai_mat act = mat_new(2, 1);
  act.bia.mem[0] = 3; /* (+) */ act.wei.mem[0] = -2; /* (*) lhs + */ act.wei.mem[1] = -2 /* (*) rhs */;
  ai_vec inp = vec_new(2);
  inp.mem[0] = lhs;
  inp.mem[1] = rhs;
  mat_mul_v(& act, inp);
  return act.out.mem[0];
}

#endif