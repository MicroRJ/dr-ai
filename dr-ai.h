/*    Copyright(C) J. Dayan Rodriguez (RJ), 2022-2023, All rights reserved.
**
** --- Todos:
**   - I think it would be good idea to have my own file format with the compressed images in a header file...
**     That way it would just be a single header file for the whole project, a rather large header file, maybe not in the header file...
**   - Much better random number generator...
**
**
** --- Note: the following are my notes but do not consider them to be entirely truthful, in fact,
**     they could be misleading if you're a novice, I am in no way an expert in this field either,
**     so consider them to be more like personal notes and not notes to be learning from.
**
**     If you are interested in checkout the source I used for this implementation please see the included reference in this
**     source directory.
**
** --- Note: The premise, and the way I look at it...
**
**     I've implemented very simple genetic algorithms in the past, these reassemble natural selection,
**     only the fittest individuals get to survive and produce offspring that is ideally, better.
**
**     If you think about how you'd emulate the aforementioned scenario, you realize you'd at least three major things.
**     First, the concept of an "individual" with some sort of potential, and then you need a way to evaluate how fit or
**     successful they are based on how they've spent that potential, then you need a way to find the best individual,
**     or individuals, if you're evaluating them based on multiple parameters - perhaps you're not just looking for the fastest, but
**     also the strongest - to then somehow reproduce them.
**
**   Now, the following algorithm, is similar in the sense that you are trying to improve or get better over time, but
**   they are entirely different in their approach.
**
**   For instance, from the genetic algorithm's perspective, there's no learning or improving for the 'individual' itself,
**   in fact, the 'individual' is screwed in that regard, it is only the offspring that introduces mutation, the genetic algorithm
**   only cares about the grand scheme of the game, who wins, it's a matter of harsh, cold probability.
**
**   However, with a neural algorithm the playing field changes, is that warm and lustrous beacon of hope that every individual innately
**   has, the only true variable and our best weapon against despair, adaptability, learning how to get 'good' and doing so in
**   a seemingly optimal manner. The 'optimal' part is also a field of research.
**
**   In other words, the genetic algorithm is the game, and the 'individual' is the neural algorithm,
**   learning how to play it.
**
**   In a less romanticized way, a neural network is a set of functions, lots of them really, laid out according
**   to which model we're implementing, you pass in an argument, you get an output, the question is, what
**   are the parameters? And furthermore, what are the right parameters?
**   And with that it becomes a little bit more obvious how this is just differential calculus, it's an
**   amalgamation of observations and research.
**
**   Our algorithm is actually quite simple, in fact, it is really simple, however, it can recognize unseen digits
**   with an accuracy of up to ~90% with just one pass in roughly less than half a second with only two layers,
**   at 60,000 28 by 28 image samples, it's about 94 million pixels per second..
**
**   The accuracy could be more or less, it just depends on how long you let it train for, the more it gets used to the database,
**   the better, but that does not mean it'll perform as good with a distinct data set.
**   So you see, it is about finding what parameters work best for all the samples, the more parameters and layers the more detail,
**   like seeing in higher resolution with depth perception, but of course, as with anything that requires computation, the more,
**   the slower.
**
** --- Note: The math.
**
**   Let's start with the basics... and I should get back to this and actually explain the basics, limits, derivatives ...
**
** -- Note: `d` is the sample index subscript
** -- Note: `k` is the layer index subscript
** -- Note: `d` is the input index subscript
** -- Note: `j` is the neuron index subscript
** -- Note: `i` is the weight index subscript
** -- Note: `r` is the neuron count subscript
**
**
** -- Note: `k[0]`: is the first layer, the output layer
**
** -- Note: `k[-1]`: is is the last layer, the input layer (negative array indices)
**
** -- Note: `k+1`: the Layer After `k` (closer to the input layer)
**
** -- Note: `k-1`: the Layer Before `k` (closer to the output layer)
**
** -- Note: `sig(x) := 1/(1 + exp(- x))` is our activation function
**
** -- Note: `sig"(x) := (1-sig(x))*sig(x)` is our activation function's derivative
**
** -- Note: `A[k:j] := sum(i, W[k:j,i] * O[k+1:j,i]) + B[k:j]` is the output of our neuron before the activation function.
**
** -- Note: `O[k:j] := sig(A[k,j])` is the output of our neuron
**
** -- Note: `Y[d:j]` is our target output, (we compare the output of a neuron in the output layer to this)
**
** -- Note: `E[d:j](X) := 1/2 * pow<2>(O[0:j] - Y[d:j])` is our cost function, (notice how it takes `O[0:j]` and not `A[0:j]`)
**
**   Let's begin with the output layer: `k = 0`
**
**   What are we trying to do?
**   Find how much each neuron of the output layer affects each `E[d:j]` so that
**   we now how much to nudge each weight and bias appropriately.
**
**   Let's find out how much `E[d:j]` is affected by the unfiltered output `A[k:j]` of a neuron.
**
**   Since `E[d:j]` is a function of `O[k:j]` and not a function of `A[k:j]` directly, we have to go through `O[k:j]`,
**   which is actually modulating `A[k:j]` through the activation function.
**
**   And so we get:
**
** 0x00: E[d:j]\\A[k:j] := E[d:j]\\O[k:j] * O[k:j]\\A[k:j]
**
**   The partial change in `E[d:j]` due to `A[k:j]` is the product of the change in `E[d:j]` due to `O[k:j]` and the change
**   in `O[k:j]` due to `A[k:j]`.
**
**   We call this term `E[d:j]\\A[k:j]` the "error", and we'll use the dollar sign to denote it.
**   This is what it looks fully resolved at the output layer:
**
** 0x00: $[k:j] := (O[k:j] - Y[k:j]) * sig"(A[k:j])
**
**   Since the error is defined per node at the output layer, we have to go over every one of its nodes `j`
**   and calculate it's error.
**
** 0x00: for(j..r[k]):
** 0x00:  ?.[j] = (O[k:j] - Y[k:j]) * sig"(A[k:j])
**
**   Now that we have the `$` term, we know how much the cost function changes due to `A[k:j]`.
**   But we need to go further, how much does `A[k:j]` change due to `B[k:j]` and every single `W[k:j,i]`.
**
**   Change In `A[k:j]` due to `B[k:j]`.
**
**   For this one we don't have to do much.
**
** 0x00: A[k:j]\\B[k:j] := $[k:j] * 1 == $[k:j];
**
**
**   Change In `A[k:j]` due to `W[k:j,i]`
**
**   This one is a little bit more involved.
**
** 0x00: A[k:j]\\W[k:j,i] := $[k:j] * O[k+1:i]
**
**   Let's expand this expression a bit, to see how it works.
**
** 0x00: A[k:0]\\W[k:  0, 0  ] := $[k:0] * O[k+1:0]
** 0x00: A[k:0]\\W[k:  0, 1  ] := $[k:0] * O[k+1:1]
** 0x00: A[k:0]\\W[k:  0, 2  ] := $[k:0] * O[k+1:2]
** 0x00: A[k:0]\\W[k:  0, 3  ] := $[k:0] * O[k+1:3]
** 0x00: A[k:0]\\W[k:  0, .. ] := $[k:0] * O[k+1:3]
** 0x00: A[k:0]\\W[k:  1, .. ] := ..
** 0x00: A[k:0]\\W[k:  2, .. ] := ..
** 0x00: A[k:0]\\W[k:  3, .. ] := ..
** 0x00: A[k:0]\\W[k: .., .. ] := ..
**
**   We see that `i` corresponds to the weight `i` of node `j` of layer `k` AND to the output of node `i` in layer `k+1`.
**   Each weight is responsible for one of the outputs of the next layer.
**
**   And so the change in `A[k=0:j=0]` due to `W[k=0:j=0,i]`, is how much `O[k+1:i]` is modulating that weight.
**
**   Here's the pseudo code that computes `A[0:j]\\W[0:j,i]`.
**
** 0x00: for(j..r[k]):
** 0x00:  for(i..r[k+1]):
** 0x00:   m[j][i] = $[k:j] * O[k+1:i]
**
**   Now, why is `O[k+1:i]` the derivative of `A[k:j]\\W[k:j,i]`?
**
**   A neuron is a first-degree polynomial:
**
** 0x00: A[k:j] =
** 0x00:   W[k:0,i=0] * O[k+1:i=0] +
** 0x00:   W[k:0,i=1] * O[k+1:i=1] +
** 0x00:   W[k:0,i=2] * O[k+1:i=2] + ...;
**
**   The only difference is, that we use the `sum()` operator instead.
**
** 0x00: A[k:0]\\W[k:0,i=0] := O[k+1:i=0]
** 0x00: A[k:0]\\W[k:0,i=1] := O[k+1:i=1]
** 0x00: A[k:0]\\W[k:0,i=2] := O[k+1:i=2]
**
**   Hidden Layers, `k > 0`:
**   ---
**
**   First let's once again establish that what we're ultimately trying to do:
**   Find out how much our weights and biases have affected `E[d:j]`.
**
**   This means computing a derivation chain that extends all the way back to the final or 'output' layer...
**   Logically, you'd cache each derivative and we can call this term '$' (the error), and "back-propagate" it...
**
**   Now, every single weight, of every single neuron, affects every single output of the previous layer.
**   So to compute `E[d:l]\\A[k:j]`, we have to also compute how much `A[k:j]` has affected `A[k-1:j=0..r[k-1]]`.
**   And notice how I used '..' to signify a range or span of expressions.
**
**   E[d:l]\\A[k:j] := sum(l..r[k-1], $[k-1:l] * A[k-1:l]\\A[k:l])
**
**   $[k:j] := sum(l..r[k-1], $[k-1:l] * A[k-1:l]\\A[k:l])
**
**
**
**   Pseudo Code Version
**
** 0x00:
**  for(j..r[k]):
**   .[i] = sum(l..r[k-1]): $[k-1:l] * W[k-1:l:j] * sig"(a[k:j])
**
** ### Change In E Due To W
** 0x00: E\\W[k:j:i] = $[k:j] * O[k+1:i]
** 0x00:
**  for(j..r[k]):
**    for(i..r[k+1]):
**     .[j][i] = $[k:j] * O[k+1:i]
** ### C Version
**  * Example A: `W[k:j,i]`
**   Visually, `k` would be above `W` and `j` below, `i` would be next to `j`.
** * Example B: `W[k[c:v]:j,i]`
**    Visually, `k` would be above `W` and `j` below, `i` would be next to `j`,
**    `c` would be above `k` and `v` below.
**   Subscript Inference:
**   Only when repetitiveness becomes obfuscating, AND if by pattern of repetition, the context
**   is unambiguous, AND it is in the best interest of the reader, it is then allowed to omit
**   the <subscript-argument> or one if its <operands>.
**   The subscript however, must be representative of the number of expressions there are in it.
** * Example C & D:
**     `w[k:j:0]` = `w[::0]`
**     `w[k]`     = `w[]`
** Let's rewrite this expression instead as:
**
** 0x00: $[k:j] := sig"(a[k:j]) * sum(l, $[k-1:l] * W[k-1:l:j])
** Let's expand this expression to see how it works:
**
** 0x00:
**  $[k:0] := sig"(a[k:0]) *
**    ( $[k-1: 0]* W[k-1 : 0: 0] +
**      $[   : 1]* W[    :  : 0] +
**      $[   : 2]* W[    :  : 0] +
**      $[   : 3]* W[    :  : 0] )
**
** Notice how the weight index is the same as j, 0. That is
** because we're only concerned with the weight that is scaling our output.
** All the other variables can be treated as 0's. {{explanation}}
**
**
** If you're somehow thinking about a matrix transposition, you're in the right path.
**   0  1  2  3  4     0  1  2
** 0 [W][W][W][W][W]   [W][W][W] 0 <
** 1 [W][W][W][W][W] T [W][W][W] 1 <
** 2 [W][W][W][W][W]   [W][W][W] 2 <
**    ^  ^  ^  ^  ^    [W][W][W] 3 <
**                     [W][W][W] 4 <
**
** To transpose a matrix is to rotate it. This could have a deeper meaning, depending on how you
** interpret it, e.i Linear Algebra, but to us, it means nothing but facilitating an operation.
**
** For the 'fast' implementation however, we're not going to do this, because transposing ~4KB worth of matrix
** every iteration sounds almost slower than Visual Studio.
** (28*28*4 + 16*10*4 + 10*10*4)/1024
** The whole point of transposing a matrix is to take advantage of SIMD, making memory that was sparse, contiguous.
** And you could make the case that transposing a matrix is a good way of doing this, and it is, if you don't know
** what you're talking about.
**
** Now, we already have SIMD working for the forward propagation stage, just not for the backwards one.
** So it's clear that we'll either have to make some sort of compromise. Let's not dwell on this too
** much though, let's write a loop instead and we'll figure something out later.
*/

#ifndef Assert
# define Assert(...) 0
#endif

// -- Todo: how to detect this automatically?
#if defined(_LANE_512)
typedef __m512 lane_t;
#elif defined(_LANE_256)
typedef __m256 lane_t;
#elif defined(_LANE_128)
typedef __m128 lane_t;
#else
typedef float lane_t;
#endif

#define lane_size_t (sizeof(lane_t)/sizeof(float))

#if defined(_LANE_512)
#define lane_load(l)         _mm512_load_ps(l)
#define lane_store(l,r)      _mm512_store_ps(l,r)
#define lane_set1(r)         _mm512_set1_ps(r)
#define lane_mul(l,r)        _mm512_mul_ps(l,r)
#define lane_div(l,r)        _mm512_div_ps(l,r)
#define lane_add(l,r)        _mm512_add_ps(l,r)
#define lane_sub(l,r)        _mm512_sub_ps(l,r)
#define lane_muladd(l,r0,r1) _mm512_fmadd_ps(r0,r1,l)
#elif defined(_LANE_256)
#define lane_load(l)         _mm256_load_ps(l)
#define lane_store(l,r)      _mm256_store_ps(l,r)
#define lane_set1(r)         _mm256_set1_ps(r)
#define lane_mul(l,r)        _mm256_mul_ps(l,r)
#define lane_div(l,r)        _mm256_div_ps(l,r)
#define lane_add(l,r)        _mm256_add_ps(l,r)
#define lane_sub(l,r)        _mm256_sub_ps(l,r)
#define lane_muladd(l,r0,r1) _mm256_fmadd_ps(r0,r1,l)
#define lane_exp(r)          _mm256_exp_ps(r)
#elif defined(_LANE_128)
#define lane_load(l)         _mm_load_ps(l)
#define lane_store(l,r)      _mm_store_ps(l,r)
#define lane_set1(r)         _mm_set1_ps(r)
#define lane_mul(l,r)        _mm_mul_ps(l,r)
#define lane_div(l,r)        _mm_div_ps(l,r)
#define lane_add(l,r)        _mm_add_ps(l,r)
#define lane_sub(l,r)        _mm_sub_ps(l,r)
#define lane_muladd(l,r0,r1) _mm_fmadd_ps(r0,r1,l)
#else
# error 'TODO(RJ)'
#endif


// -- Note: store vectors of arbitrary length but that are allocated efficiently, so padding may be present...
typedef struct vector_t vector_t;
typedef struct vector_t
{ int      len, max;
  float  * mem;
} vector_t;

typedef struct matrix_t matrix_t;
typedef struct matrix_t
{ int     min, max;
  float * mem;
  int     vec_min, vec_max;
  int     col, row;
} matrix_t;

typedef struct layer_t layer_t;
typedef struct layer_t
{ matrix_t wei;
  vector_t bia;
  union
  { vector_t new_bia;
    vector_t err;
  };
  vector_t tmp;
  matrix_t new_wei;
  vector_t out;
  vector_t act;
} layer_t;

typedef struct network_t network_t;
typedef struct network_t
{ layer_t   lay_o;
  layer_t   lay_i;
  layer_t   layer[0x10];
  float     alpha;
} network_t;

// --- Note: we don't actually use this because there's a more efficient way ...

// https://www.desmos.com/calculator/btdhngc1oq
#define sigmoid(o) (1.0/(1.0+exp(-o)))
#define sigmoid_prime(o) (sigmoid(o) * (1 - sigmoid(o)))

ccfunc ccinle float square_real32(float val);
ccfunc ccinle float cube_real32(float val);

ccfunc ccinle unsigned int xorshift32(unsigned int x);
ccfunc ccinle double xorshift_randreal64(double min, double max);
ccfunc ccinle float  xorshift_randreal32(float min, float max);

ccfunc ccinle void vec_mul(int, float *, float *, float *);
ccfunc ccinle void vec_div(int, float *, float *, float *);
ccfunc ccinle void vec_add(int, float *, float *, float *);
ccfunc ccinle void vec_sub(int, float *, float *, float *);
ccfunc ccinle void vec_muladd(int, float *, float *, float *, float alpha);
ccfunc ccinle void vec_dot(int, float *, float *, float *);
ccfunc ccinle void vec_zro(int, float *);


ccfunc ccinle float square_real32(float val)
{
  return val*val;
}

ccfunc ccinle float cube_real32(float val)
{
  return val*val*val;
}

ccfunc ccinle unsigned int xorshift32(unsigned int x)
{ x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

// --- Todo: this isn't proper, see https://prng.di.unimi.it ...
ccfunc ccinle float xorshift_randreal32(float min, float max)
{ ccglobal ccthread_local unsigned int state = 901823102;
  state = xorshift32(state);

  double v=(double)state/~0u;
  return (float)(min+(max-min)*v);
}

ccfunc ccinle void
vec_mul(int len, float *dst, float *lhs, float *rhs)
{
  for(int i=0; i<len; i+=lane_size_t)
    lane_store(dst+i,lane_mul(lane_load(lhs+i),lane_load(rhs+i)));
}

ccfunc ccinle void
vec_div(int len, float *dst, float *lhs, float *rhs)
{
  for(int i=0; i<len; i+=lane_size_t)
    lane_store(dst+i,lane_div(lane_load(lhs+i),lane_load(rhs+i)));
}

ccfunc ccinle void
vec_add(int len, float *dst, float *lhs, float *rhs)
{
  for(int i=0; i<len; i+=lane_size_t)
    lane_store(dst+i,lane_add(lane_load(lhs+i),lane_load(rhs+i)));
}

ccfunc ccinle void
vec_sub(int len, float *dst, float *lhs, float *rhs)
{
  for(int i=0; i<len; i+=lane_size_t)
    lane_store(dst+i,lane_sub(lane_load(lhs+i),lane_load(rhs+i)));
}

ccfunc ccinle void
vec_muladd(int len, float *dst, float *lhs, float *rhs, float alpha)
{
  for(int i=0; i<len; i+=lane_size_t)
    lane_store(dst+i,lane_muladd(lane_load(lhs+i),lane_load(rhs+i),lane_set1(alpha)));
}


ccfunc ccinle void
vec_dot(int len, float *dst, float *lhs, float *rhs)
{
  // --- Note: not sure if we're at the point in which we'd want multiple accumulators for higher precision.
  lane_t acc=lane_set1(0);

  // --- Todo: why can't I keep the result loaded in a register and keep accumulating?
  for(int i=0; i<len; i+=lane_size_t)
    lane_store((float *)&acc,lane_muladd(acc,lane_load(lhs+i),lane_load(rhs+i)));

  // -- Note: clearly, you could also make this SIMD.
  float v=0.f;
  for(int i=0;i<lane_size_t;i+=1)
    v+=((float*)&acc)[i];
  *dst=v;
}

ccfunc ccinle void
vec_zro(int len, float *dst)
{
  lane_t set=lane_set1(0.f);
  for(int i=0; i<len; i+=lane_size_t)
    lane_store(dst+i,set);
}

// -- Todo: this could be made so much faster ...
ccfunc ccinle void
vec_rnd(int len, float *dst)
{
  for(int i=0; i<len; ++i)
    dst[i]=xorshift_randreal32(-1.f,+1.f);
}

ccfunc ccinle int
vec_max(int len)
{
  return (lane_size_t-1+len)/lane_size_t*lane_size_t;
}

#ifndef del_vec
# define del_vec(vec) _aligned_free(vec.mem);
#endif

static __forceinline vector_t new_vec(int len)
{
  int  max = vec_max(len);
  // TODO(RJ):
  // ; this is temporary!
  void  * mem = _aligned_malloc(sizeof(float) * max, 0x20);
  // Todo: vec-zero
  memset(mem, 0x00, sizeof(float) * max);

  vector_t v;
  v.len = len;
  v.max = max;
  v.mem = (float *) mem;
  return v;
}

// I wonder what template++ ppl are thinking right now...
static vector_t new_vec10(
  float s0, float s1,
  float s2, float s3,
  float s4, float s5,
  float s6, float s7,
  float s8, float s9 )
{ vector_t v = new_vec(0x0A);
  v.mem[0x00] = s0; v.mem[0x01] = s1;
  v.mem[0x02] = s2; v.mem[0x03] = s3;
  v.mem[0x04] = s4; v.mem[0x05] = s5;
  v.mem[0x06] = s6; v.mem[0x07] = s7;
  v.mem[0x08] = s8; v.mem[0x09] = s9;
  return v;
}

// Create a new matrix.
static matrix_t new_mat_(int col, int row, int min, int vec)
{
// For a col matrix, the number of vectors has to the be the number of columns
// and the vector size has to be the number of rows.
#ifndef new_col_mat
# define new_col_mat(col,row) new_mat_(col,row,col,row)
#endif
// For a row matrix, the number of vectors has to the be the number of rows
// and the vector size has to be the number of columns.
#ifndef new_row_mat
# define new_row_mat(col,row) new_mat_(col,row,row,col)
#endif

#ifndef rpl_mat
# define rpl_mat(mat) new_mat_(mat.col,mat.row,mat.min,mat.vec_min)
#endif

  matrix_t m;
  m.col     = col;
  m.row     = row;
  m.min     = min;
  m.max     = min;
  m.vec_min = vec;
  m.vec_max = vec_max(vec);

  m.mem = (float *) _aligned_malloc(sizeof(float) * m.vec_max * min, 0x20);
  memset(m.mem, 0, sizeof(float) * m.vec_max * m.min);
  return m;
}

static vector_t mat_vec(matrix_t mat, int index)
{ vector_t v;
  v.mem = mat.mem + mat.vec_max * index;
  v.len = mat.vec_min;
  v.max = mat.vec_max;
  return v;
}

ccfunc layer_t
new_lay(int col_len, int row_len)
{ layer_t n;

  n.wei     = new_row_mat(col_len,row_len);
  n.act     = new_vec(row_len);
  n.out     = new_vec(row_len);
  n.bia     = new_vec(row_len);
  n.err     = new_vec(row_len);
  n.tmp     = new_vec(row_len);
  n.new_wei = rpl_mat(n.wei);

  vec_rnd(n.wei.min*n.wei.vec_max,n.wei.mem);
  vec_rnd(n.bia.len,n.bia.mem);
  return n;
}

ccfunc ccinle vector_t
layer_feed(layer_t *layer, vector_t x)
{
  matrix_t w=layer->wei;
  vector_t o=layer->out;
  vector_t b=layer->bia;
  vector_t a=layer->act;

  ccassert(x.len==w.col);
  ccassert(o.len==w.row);
  ccassert(b.len==w.row);

  int i;

  for(i=0;i<w.row;++i)
    vec_dot(x.len,o.mem+i,w.mem+w.vec_max*i,x.mem);

  vec_add(o.len,o.mem,o.mem,b.mem);

  ccassert(o.len==a.len);

  // -- Note: We could have a callback here for custom activation functions...
  lane_t n=lane_set1(1);
  lane_t z=lane_set1(0);
  for(i=0; i<o.len; i+=lane_size_t)
    lane_store(a.mem+i,
      // -- Note: apply the activation function, sigmoid..
      lane_div(n,lane_add(n,lane_exp(lane_sub(z,lane_load(o.mem+i))))));
  return a;
}

ccfunc ccinle void
layer_update(layer_t *lay, float alpha)
{
  vector_t bia_old=lay->bia,bia_new=lay->new_bia;
  vec_muladd(bia_old.len,bia_old.mem,bia_old.mem,bia_new.mem,-alpha);

  matrix_t wei_old=lay->wei,wei_new=lay->new_wei;
  vec_muladd(wei_old.min*wei_old.vec_max,wei_old.mem,wei_old.mem,wei_new.mem,-alpha);
}

ccfunc ccinle void
network_init(network_t *net, int inp, int con, int out)
{ net->lay_i=new_lay(inp,con);
  net->lay_o=new_lay(con,out);
}

ccfunc ccinle void
network_feed(network_t *net, vector_t inp)
{
  // -- Todo: feed all the layers you dude...
  layer_feed(& net->lay_o,
    layer_feed(& net->lay_i, inp));
}

ccfunc ccinle void
network_reverse_feed(network_t * net, vector_t inp_v, vector_t tar_v)
{
  layer_t
    lay_o = net->lay_o,
    lay_i = net->lay_i;
  vector_t
    out_o = lay_o.out,
    act_o = lay_o.act,
    err_o = lay_o.err,
    tmp_o = lay_o.tmp,

    out_i = lay_i.out,
    act_i = lay_i.act,
    err_i = lay_i.err,
    tmp_i = lay_i.tmp;
  matrix_t
    wei_o     = lay_o.wei,
    wei_i     = lay_i.wei,
    new_wei_o = lay_o.new_wei,
    new_wei_i = lay_i.new_wei;

  ccassert(inp_v.len==wei_i.col);

  int row,col,i;
  float acc;

  lane_t n=lane_set1(1);
  lane_t z=lane_set1(0);

  // -- Todo: explain why this calculation is simpler than on the other layers..
  for(i=0; i<err_o.len; i+=lane_size_t)
  { lane_t a=lane_load(act_o.mem+i);
    lane_t y=lane_load(tar_v.mem+i);
    lane_store(err_o.mem+i,
      lane_mul(lane_sub(a,y),
        // -- Note: calculate sigmoid prime...
        lane_mul(lane_sub(n,a),a)));
  }

  // -- Todo: speed!
  for(row=0;row<wei_o.row;++row)
    for(col=0;col<wei_o.col;++col)
      new_wei_o.mem[row*new_wei_o.vec_max+col]=err_o.mem[row]*act_i.mem[col];

  // -- Todo: revise this.. there's possibly something wrong with it...
  for(i=0; i<out_i.len; i+=lane_size_t)
  {
    // -- Note: why were we doing this? report to Jacob?..
    // lane_t a=lane_div(n,lane_add(n,lane_exp(lane_sub(z,lane_load(out_i.mem+i)))));
    // lane_t d=lane_mul(lane_sub(n,a),a);

    lane_t a=lane_load(act_i.mem+i);
    lane_store(err_i.mem+i,
      // -- Note: calculate sigmoid prime..
      lane_mul(lane_sub(n,a),a));
  }

  // -- Note: this is so you see how they link together..
  ccassert(wei_i.row==wei_o.col);
  ccassert(wei_o.col==wei_i.row);

  // -- Todo: I can't fathom a transpose being faster here, we have to test it out..
  for(row=0;row<wei_i.row;++row)
  { acc=0;
    for(col=0;col<wei_o.row;++col)
      acc+=err_o.mem[col]*wei_o.mem[col*wei_o.vec_max+row];
    err_i.mem[row]*=acc;
  }

  // -- Todo: speed!
  for(row=0;row<wei_i.row;++row)
    for(col=0;col<wei_i.col;++col)
      new_wei_i.mem[row*new_wei_i.vec_max+col]=err_i.mem[row]*inp_v.mem[col];
}

ccfunc ccinle void
network_train(network_t *network, vector_t i, vector_t t)
{
  // --- Todo: actually reset all the layers..
  vec_zro(network->lay_o.new_bia.len,network->lay_o.new_bia.mem);
  vec_zro(network->lay_i.new_bia.len,network->lay_i.new_bia.mem);

  vec_zro(network->lay_o.new_wei.min*network->lay_o.new_wei.vec_max,network->lay_o.new_wei.mem);
  vec_zro(network->lay_i.new_wei.min*network->lay_i.new_wei.vec_max,network->lay_i.new_wei.mem);

  network_feed(network,i);
  network_reverse_feed(network,i,t);

  // --- Todo: actually update all the layers..
  layer_update(&network->lay_i,.1f);
  layer_update(&network->lay_o,.1f);
}

static int network_prediction(network_t *net)
{ float prd_val=-1;
  int   prd_idx=-1;
  for(int i = 0; i < net->lay_o.act.len; ++ i)
  { if(net->lay_o.act.mem[i] > prd_val)
    { prd_val = net->lay_o.act.mem[i];
      prd_idx = i;
    }
  }
  return prd_idx;
}
