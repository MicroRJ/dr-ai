#ifndef DR_LAN
#define DR_LAN

typedef float drl__f32;

#if defined(DRL_512)
# define drl__lane __m512
#elif defined(DRL_256)
# define drl__lane __m256
#elif defined(DRL_128)
# define drl__lane __m128
#else
# define drl__lane drl__f32
#endif

#define drl__size (sizeof(drl__lane)/sizeof(drl__f32))

// drls__pull(src)
// drls__push(dst,opr)
// drls__set1(opr)
// drls__pow(dst,lhs,rhs)
// drls__exp(lhs,rhs)
// drls__mul(lhs,rhs)
// drls__div(lhs,rhs)
// drls__add(lhs,rhs)
// drls__sub(lhs,rhs)
// drls__mad(dst,lhs,rhs)
// drls__sig(rhs)
// drls__dsg(rhs)

#if defined(DRL_512)
#define drls__pull(src)        _mm512_load_ps(src)
#define drls__push(dst,opr)    _mm512_store_ps(dst,opr)
#define drls__set1(opr)        _mm512_set1_ps(opr)
#define drls__pow(lhs,rhs)     _mm512_pow_ps(lhs,rhs)
#define drls__exp(lhs,rhs)     _mm512_exp_ps(lhs,rhs)
#define drls__mul(lhs,rhs)     _mm512_mul_ps(lhs,rhs)
#define drls__div(lhs,rhs)     _mm512_div_ps(lhs,rhs)
#define drls__add(lhs,rhs)     _mm512_add_ps(lhs,rhs)
#define drls__sub(lhs,rhs)     _mm512_sub_ps(lhs,rhs)
#define drls__mad(dst,lhs,rhs) _mm512_fmadd_ps(lhs,rhs,dst)
#elif defined(DRL_256)
#define drls__pull(src)        _mm256_load_ps(src)
#define drls__push(dst,opr)    _mm256_store_ps(dst,opr)
#define drls__set1(opr)        _mm256_set1_ps(opr)
#define drls__pow(lhs,rhs)     _mm256_pow_ps(lhs,rhs)
#define drls__exp(lhs,rhs)     _mm256_exp_ps(lhs,rhs)
#define drls__mul(lhs,rhs)     _mm256_mul_ps(lhs,rhs)
#define drls__div(lhs,rhs)     _mm256_div_ps(lhs,rhs)
#define drls__add(lhs,rhs)     _mm256_add_ps(lhs,rhs)
#define drls__sub(lhs,rhs)     _mm256_sub_ps(lhs,rhs)
#define drls__mad(dst,lhs,rhs) _mm256_fmadd_ps(lhs,rhs,dst)
#elif defined(DRL_128)
#define drls__pull(src)        _mm_load_ps(src)
#define drls__push(dst,opr)    _mm_store_ps(dst,opr)
#define drls__set1(opr)        _mm_set1_ps(opr)
#define drls__pow(lhs,rhs)     _mm_pow_ps(lhs,rhs)
#define drls__exp(lhs,rhs)     _mm_exp_ps(lhs,rhs)
#define drls__mul(lhs,rhs)     _mm_mul_ps(lhs,rhs)
#define drls__div(lhs,rhs)     _mm_div_ps(lhs,rhs)
#define drls__add(lhs,rhs)     _mm_add_ps(lhs,rhs)
#define drls__sub(lhs,rhs)     _mm_sub_ps(lhs,rhs)
#define drls__mad(dst,lhs,rhs) _mm_fmadd_ps(lhs,rhs,dst)
#else
# error 'TODO(RJ)'
#endif


#if defined(drl__lane)
# define drls__neg(lan) drls__sub(drl__zro, lan)

# define drls__sig(lan) drls__div(drls__set1(1.f), drls__add(drls__set1(1.f), drls__exp(drls__neg(lan))))
# define drls__dsg(lan) drls__mul(drls__sub(drls__set1(1.f),lan), lan)
#endif


static void __forceinline drvs__pow(unsigned int len, drl__f32 *dst, drl__f32 *lhs, drl__f32 *rhs)
{ for(unsigned int i = 0; i < len; i += drl__size)
  { drls__push(dst + i, drls__pow(drls__pull(lhs + i),drls__pull(rhs + i)));
  }
}
static void __forceinline drvs__exp(unsigned int len, drl__f32 *dst, drl__f32 *lhs, drl__f32 *rhs)
{ for(unsigned int i = 0; i < len; i += drl__size)
  { // drls__push(dst + i, drls__exp(drls__pull(lhs + i),drls__pull(rhs + i)));
  }
}
static void __forceinline drvs__mul(unsigned int len, drl__f32 *dst, drl__f32 *lhs, drl__f32 *rhs)
{ for(unsigned int i = 0; i < len; i += drl__size)
  { drls__push(dst + i, drls__mul(drls__pull(lhs + i),drls__pull(rhs + i)));
  }
}
// dst = lhs + rhs * alpha
// 3:3 ratio
static void __forceinline drvs__maa(unsigned int len, drl__f32 *dst, drl__f32 *lhs, drl__f32 *rhs, drl__f32 alpha)
{
  for(unsigned int i = 0; i < len; i += drl__size)
  { drl__lane lhs_l = drls__pull(lhs + i);
    drl__lane rhs_l = drls__pull(rhs + i);
    drls__push(dst + i, drls__mad(lhs_l, rhs_l, drls__set1(alpha)));
  }
}
static void __forceinline drvs__div(unsigned int len, drl__f32 *dst, drl__f32 *lhs, drl__f32 *rhs)
{ for(unsigned int i = 0; i < len; i += drl__size)
  { drls__push(dst + i, drls__div(drls__pull(lhs + i),drls__pull(rhs + i)));
  }
}
static void __forceinline drvs__sub(unsigned int len, drl__f32 *dst, drl__f32 *lhs, drl__f32 *rhs)
{ for(unsigned int i = 0; i < len; i += drl__size)
  { drls__push(dst + i, drls__sub(drls__pull(lhs + i),drls__pull(rhs + i)));
  }
}
static void __forceinline drvs__add(unsigned int len, drl__f32 *dst, drl__f32 *lhs, drl__f32 *rhs)
{ for(unsigned int i = 0; i < len; i += drl__size)
  { drls__push(dst + i, drls__add(drls__pull(lhs + i),drls__pull(rhs + i)));
  }
}
// Not sure if we're at the point in which we'd want multiple accumulators for higher precision.
static void __forceinline drvs__dot(unsigned int len, drl__f32 *dst, drl__f32 *lhs, drl__f32 *rhs)
{ drl__lane acc = {};
  for(unsigned int i = 0; i < len; i += drl__size)
  { drls__push((drl__f32 *)&acc, drls__mad(acc, drls__pull(lhs + i),drls__pull(rhs + i)));
  }

  // Clearly, you could also make this SIMD.
  drl__f32 dot = 0.f;
  for(unsigned int i = 0; i < drl__size; i += 1)
  { dot += ((drl__f32 *) &acc)[i];
  }
  *dst = dot;
}
static void __forceinline drvs__mov(unsigned int len, drl__f32 *dst, drl__f32 *lhs)
{ for(unsigned int i = 0; i < len; i += drl__size)
  { drls__push(dst + i, drls__pull(lhs + i));
  }
}
static void __forceinline drvs__one(unsigned int len, drl__f32 *dst)
{ for(unsigned int i = 0; i < len; i += drl__size)
  { drls__push(dst + i, drls__set1(1.f));
  }
}
static void __forceinline drvs__zro(unsigned int len, drl__f32 *dst)
{ for(unsigned int i = 0; i < len; i += drl__size)
  { drls__push(dst + i, drls__set1(0.f));
  }
}


// You can't really improve this much, 3 seperate drvs calls do about the same
// damage. Difference is that this method doesn't use extra memory.
// err = (act - tar) * ((1 - tar) * tar)
static void __forceinline drvs__aii_err_out(
  unsigned int   len,
  drl__f32     * err,
  drl__f32     * tar,
  drl__f32     * act )
{ for(unsigned int i = 0; i < len; i += drl__size)
  { drls__push(err + i,
      drls__mul(
        drls__sub(
          drls__pull(act + i),
          drls__pull(tar + i)),
        drls__dsg(
          drls__pull(act + i))));
  }
}

#endif