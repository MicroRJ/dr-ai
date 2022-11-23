#ifndef DR_VEC
#define DR_VEC


typedef float ai_float;
typedef int   ai_int;


#if defined(__AVX2__)
# define ai_lane                      __m512
# define ai_lane_load(tar)            _mm512_load_ps(tar)
# define ai_lane_store(lhs,rhs)       _mm512_store_ps(lhs, rhs)
# define ai_lane_exp(lhs,rhs)         _mm512_exp_ps(lhs, rhs)
# define ai_lane_mul(lhs,rhs)         _mm512_mul_ps(lhs, rhs)
# define ai_lane_div(lhs,rhs)         _mm512_div_ps(lhs, rhs)
# define ai_lane_add(lhs,rhs)         _mm512_add_ps(lhs, rhs)
# define ai_lane_sub(lhs,rhs)         _mm512_sub_ps(lhs, rhs)
# define ai_lane_mul_add(dst,lhs,rhs) _mm512_fmadd_ps(lhs,rhs,dst)
#define ai_lane_collapse_additive(l)\
  (l.m256_f32[0x00] + l.m256_f32[0x01] + l.m256_f32[0x02] + l.m256_f32[0x03] +\
   l.m256_f32[0x04] + l.m256_f32[0x05] + l.m256_f32[0x06] + l.m256_f32[0x07] +
   l.m256_f32[0x08] + l.m256_f32[0x09] + l.m256_f32[0x0A] + l.m256_f32[0x0B] +\
   l.m256_f32[0x0C] + l.m256_f32[0x0D] + l.m256_f32[0x0E] + l.m256_f32[0x0F])
#elif defined(__AVX__) // <-- this is wrong! 256 instructions are not here, they are above!
# define ai_lane                       __m256
# define ai_lane_load(tar)             _mm256_load_ps(tar)
# define ai_lane_store(lhs,rhs)        _mm256_store_ps(lhs, rhs)
# define ai_lane_exp(lhs)              _mm256_exp_ps(lhs)
# define ai_lane_mul(lhs,rhs)          _mm256_mul_ps(lhs, rhs)
# define ai_lane_div(lhs,rhs)          _mm256_div_ps(lhs, rhs)
# define ai_lane_add(lhs,rhs)          _mm256_add_ps(lhs, rhs)
# define ai_lane_sub(lhs,rhs)          _mm256_sub_ps(lhs, rhs)
# define ai_lane_mul_add(dst,lhs,rhs)  _mm256_fmadd_ps(lhs,rhs,dst)
#define ai_lane_collapse_additive(l)\
  (l.m256_f32[0x00] + l.m256_f32[0x01] + l.m256_f32[0x02] + l.m256_f32[0x03] +\
   l.m256_f32[0x04] + l.m256_f32[0x05] + l.m256_f32[0x06] + l.m256_f32[0x07])
#elif defined(__SSE__)
# define ai_lane                      __m128
# define ai_lane_load(lhs,rhs)        _mm_load_ps(lhs, rhs)
# define ai_lane_store(lhs,rhs)       _mm_store_ps(lhs, rhs)
# define ai_lane_exp(lhs,rhs)         _mm_exp_ps(lhs, rhs)
# define ai_lane_mul(lhs,rhs)         _mm_mul_ps(lhs, rhs)
# define ai_lane_div(lhs,rhs)         _mm_div_ps(lhs, rhs)
# define ai_lane_add(lhs,rhs)         _mm_add_ps(lhs, rhs)
# define ai_lane_sub(lhs,rhs)         _mm_sub_ps(lhs, rhs)
# define ai_lane_mul_add(dst,lhs,rhs) _mm_fmadd_ps(lhs,rhs,dst)
#define ai_lane_collapse_additive(l)\
  (l.m128_f32[0x00] + l.m128_f32[0x01] + l.m128_f32[0x02] + l.m128_f32[0x03])
#else
// Welp...
#endif


typedef struct ai_vec
{ ai_int     len, max;
  ai_float * mem;
} ai_vec;

typedef struct ai_mat
{ ai_int     col, row;
  ai_int     vec_min, vec_max;
  ai_int     min, max;
  ai_float * mem;
} ai_mat;

// TODO(RJ): make this a wide instruction too!
/* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
unsigned int xorshift32(unsigned int x)
{ x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

ai_float rand_f(ai_float min, ai_float max)
{ static unsigned int state = xorshift32(7); // TODO(RJ)!
  return min + (ai_float)state/~0u * (max - min);
}

static ai_int __forceinline vec_max(ai_int len)
{
#if defined(ai_lane)
  const ai_int lan = sizeof(ai_lane)/sizeof(ai_float);
  return (lan-1 + len) / lan * lan;
#else
  return len;
#endif
}

#ifndef del_vec
# define del_vec(vec) _aligned_free(vec.mem);
#endif

#ifndef rpl_vec
# define rpl_vec(vec) new_vec(vec.len)
#endif

static __forceinline ai_vec new_vec(ai_int len)
{
  ai_int  max = vec_max(len);
  // TODO(RJ):
  // ; this is temporary!
  void  * mem = _aligned_malloc(sizeof(ai_float) * max, 0x20);
  memset(mem, 0x00, sizeof(ai_float) * max);

  ai_vec v;
  v.len = len;
  v.max = max;
  v.mem = (ai_float *) mem;
  return v;
}

static ai_vec new_vec2(
  ai_float s0, ai_float s1 )
{ ai_vec v = new_vec(0x02);
  v.mem[0x00] = s0; v.mem[0x01] = s1;
  return v;
}

static ai_vec new_vec3(
  ai_float s0,
  ai_float s1,
  ai_float s2 )
{ ai_vec v = new_vec(0x03);
  v.mem[0x00] = s0;
  v.mem[0x01] = s1;
  v.mem[0x02] = s2;
  return v;
}

static ai_vec new_vec4(
  ai_float s0, ai_float s1,
  ai_float s2, ai_float s3 )
{ ai_vec v = new_vec(0x04);
  v.mem[0x00] = s0; v.mem[0x01] = s1;
  v.mem[0x02] = s2; v.mem[0x03] = s3;
  return v;
}

// I wonder what template++ ppl are thinking right now...
static ai_vec new_vec10(
  ai_float s0, ai_float s1,
  ai_float s2, ai_float s3,
  ai_float s4, ai_float s5,
  ai_float s6, ai_float s7,
  ai_float s8, ai_float s9 )
{ ai_vec v = new_vec(0x0A);
  v.mem[0x00] = s0; v.mem[0x01] = s1;
  v.mem[0x02] = s2; v.mem[0x03] = s3;
  v.mem[0x04] = s4; v.mem[0x05] = s5;
  v.mem[0x06] = s6; v.mem[0x07] = s7;
  v.mem[0x08] = s8; v.mem[0x09] = s9;
  return v;
}



// TODO(RJ): optimize
static __forceinline ai_int vec_eql(ai_vec lhs, ai_vec rhs)
{ if(lhs.len != rhs.len)
    return false;
  for(ai_int i = 0; i < lhs.len; ++ i)
    if(lhs.mem[i] != rhs.mem[i])
      return false;
  return true;
}


static __forceinline void vec_rnd(ai_vec v)
{ for(int i = 0; i < v.len; ++ i)
    v.mem[i] = rand_f(0.f, 1.f);
}

static void __forceinline __vectorcall vec_mov(ai_vec dst, ai_vec lhs)
{
#if defined(ai_lane)
  for(int i = 0; i < dst.len; i += sizeof(ai_lane)/sizeof(ai_float))
  { ai_lane_store(dst.mem+i,ai_lane_load(lhs.mem+i));
  }
#else
  memcpy(dst.mem,lhs.mem,dst.len*sizeof(* dst.mem));
#endif
}

static ai_vec __forceinline __vectorcall vec_cln(ai_vec lhs)
{
  ai_vec dst = new_vec(lhs.len);
  vec_mov(dst,lhs);
  return dst;
}


static ai_vec __forceinline __vectorcall vec_sub_(ai_vec dst, ai_vec lhs, ai_vec rhs)
{
#ifndef vec_sub
# define vec_sub(lhs,rhs) vec_sub(vec_rpl(lhs),lhs,rhs)
#endif
  Assert( lhs.len == rhs.len );
#if defined(ai_lane)
  for(int i = 0; i < dst.len; i += sizeof(ai_lane)/sizeof(ai_float))
  { ai_lane_store(dst.mem+i,ai_lane_sub(ai_lane_load(lhs.mem+i),ai_lane_load(rhs.mem+i)));
  }
#else
  for(int i = 0; i < dst.len; ++ i)
  { dst.mem[i] = lhs.mem[i] - rhs.mem[i];
  }
#endif
  return dst;
}

static ai_vec __forceinline __vectorcall vec_add_(ai_vec dst, ai_vec lhs, ai_vec rhs)
{
#ifndef vec_add
# define vec_add(lhs,rhs) vec_add_(rpl_vec(lhs),lhs,rhs)
#endif

  Assert( lhs.len == rhs.len );

#if defined(ai_lane)
  for(ai_int i = 0; i < dst.len; i += sizeof(ai_lane)/sizeof(ai_float))
  { ai_lane_store(dst.mem+i,ai_lane_add(ai_lane_load(lhs.mem+i),ai_lane_load(rhs.mem+i)));
  }
#else
  for(ai_int i = 0; i < dst.len; ++ i)
  { dst.mem[i] = lhs.mem[i] + rhs.mem[i];
  }
#endif
  return dst;
}

static ai_vec __forceinline __vectorcall vec_mul_(ai_vec dst, ai_vec lhs, ai_vec rhs)
{
#ifndef vec_mul
# define vec_mul(lhs, rhs) vec_mul_(rpl_vec(lhs),lhs,rhs)
#endif

  Assert( lhs.len == rhs.len );
#if defined(ai_lane)
  for(int i = 0; i < dst.len; i += sizeof(ai_lane)/sizeof(ai_float))
  { ai_lane_store(dst.mem+i,ai_lane_mul(ai_lane_load(lhs.mem+i),ai_lane_load(rhs.mem+i)));
  }
#else
  for(int i = 0; i < dst.len; ++ i)
  { dst.mem[i] = lhs.mem[i] * rhs.mem[i];
  }
#endif
  return dst;
}

#if defined(ai_lane)

static const ai_lane lan_one = {1,1,1,1,1,1,1,1}; // TODO(RJ)!
static const ai_lane lan_zro = {0,0,0,0,0,0,0,0}; // TODO(RJ)!

#ifndef lan_sig
# define lan_sig(lan) ai_lane_div(lan_one, ai_lane_add(lan_one, ai_lane_exp(ai_lane_sub(lan_zro,lan))))
#endif

#ifndef lan_dsg
# define lan_dsg(lan) ai_lane_mul(lan_sig(lan), ai_lane_sub(lan_one, ai_lane_exp(lan)))
#endif

#endif

static ai_vec __forceinline __vectorcall vec_sig_(ai_vec dst, ai_vec lhs)
{
  ai_int      len = dst.len;
  ai_float *mem_d = dst.mem;
  ai_float *mem_l = lhs.mem;

#if defined(ai_lane)
  for(ai_int idx = 0; idx < len; idx += sizeof(ai_lane)/sizeof(ai_float))
  { const ai_lane
      lhs_l = ai_lane_load(mem_l + idx),
      sig_l = lan_sig(lhs_l);
    ai_lane_store(mem_d + idx, sig_l);
  }
#else
  const ai_float ai_float_one = 1.f;
  for(int i = 0; i < dst.len; ++ i)
  { dst.mem[i] = ai_float_one / (ai_float_one + exp(- lhs.mem[i]));
  }
#endif
  return dst;
}

static ai_vec __forceinline __vectorcall vec_dsg_(ai_vec dst, ai_vec lhs)
{
#if defined(ai_lane)
  for(ai_int idx = 0; idx < dst.len; idx += sizeof(ai_lane)/sizeof(ai_float))
  { const ai_lane
      lhs_l = ai_lane_load(lhs.mem + idx),
      dsg_l = lan_dsg(lhs_l);
    ai_lane_store(dst.mem + idx, dsg_l);
  }
#else
  for(int i = 0; i < dst.len; ++ i)
  { dst.mem[i] = (1.f - exp(+ lhs.mem[i]) * (1.f / (1.f + exp(- lhs.mem[i]))));
  }
#endif
  return dst;
}

// Not sure if we're at the point in which we'd want multiple accumulators for higher precision.
static ai_float __forceinline __vectorcall vec_dot(ai_vec lhs, ai_vec rhs)
{ Assert(lhs.len == rhs.len);

#if defined(ai_lane)
  ai_lane lane_acc = { };
  for(ai_int i = 0; i < lhs.len; i += sizeof(ai_lane)/sizeof(ai_float))
  { ai_lane lhs_l = ai_lane_load(lhs.mem + i);
    ai_lane rhs_l = ai_lane_load(rhs.mem + i);
    lane_acc = ai_lane_mul_add( lane_acc, lhs_l, rhs_l);
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

#ifndef new_col_mat
# define new_col_mat(col,row) new_mat_(col,row,col,row)
#endif
#ifndef new_row_mat
# define new_row_mat(col,row) new_mat_(col,row,row,col)
#endif
#ifndef rpl_mat
# define rpl_mat(mat) new_mat_(mat.col,mat.row,mat.min,mat.vec_min)
#endif

static ai_mat new_mat_(ai_int col, ai_int row, ai_int min, ai_int vec)
{ ai_mat m;
  m.col = col;
  m.row = row;
  m.min = min;
  m.max = min;
  m.vec_min = vec;
  m.vec_max = vec_max(vec);

  if(m.vec_max != m.vec_min)
  { TRACE_W("matrix will be padded, natural vec size will be %i, this is for optimization purposes", m.vec_max);
  }

  m.mem = (ai_float *) _aligned_malloc((sizeof(ai_float)*m.vec_max) * min, 0x20);
  memset(m.mem,0,sizeof(ai_float) * m.vec_max);
  return m;
}

static ai_vec mat_vec(ai_mat mat, ai_int index)
{ ai_vec v;
  v.mem = mat.mem + mat.vec_max * index;
  v.len = mat.vec_min;
  return v;
}

static void mat_rnd(ai_mat mat)
{ // SPEED(RJ):
  for(ai_int idx = 0; idx < mat.min; ++ idx)
  { vec_rnd(mat_vec(mat,idx));
  }
}

static __forceinline void __vectorcall mat_dot_(ai_vec dst, ai_mat mat, ai_vec src)
{
#ifndef mat_dot
# define mat_dot(mat,src) mat_dot_(vec_rpl(src),mat,src)
#endif
  for(ai_int idx = 0; idx < mat.min; ++ idx)
  { dst.mem[idx] = vec_dot(mat_vec(mat,idx),src);
  }
}

#endif