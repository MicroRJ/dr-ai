/*
** Copyright(C) J. Dayan Rodriguez (RJ), 2022-2023, All rights reserved.
*/
#ifndef _DRAI
#define _DRAI

# include <immintrin.h>
# include <emmintrin.h>
# include    <intrin.h>

#ifndef ccfunc
# define ccfunc static
#endif
#ifndef ccinle
# define ccinle __forceinline
#endif

#if defined(_LANE_512)
typedef __m512 lane_t;
#elif defined(_LANE_256)
typedef __m256 lane_t;
#elif defined(_LANE_128)
typedef __m128 lane_t;
#else
typedef float lane_t;
#endif

#ifndef _LOOP_UNROLL
# define _LOOP_UNROLL 4
#endif
#ifndef _LANE_SIZE
# define _LANE_SIZE (sizeof(lane_t)/sizeof(float))
#endif

#if defined(_LANE_512)
# define lane_load(l)         _mm512_load_ps(l)
# define lane_store(l,r)      _mm512_store_ps(l,r)
# define lane_widen(r)        _mm512_set1_ps(r)
# define lane_mul(l,r)        _mm512_mul_ps(l,r)
# define lane_div(l,r)        _mm512_div_ps(l,r)
# define lane_add(l,r)        _mm512_add_ps(l,r)
# define lane_sub(l,r)        _mm512_sub_ps(l,r)
# define lane_muladd(l,r0,r1) _mm512_fmadd_ps(r0,r1,l)
# define lane_exp(r)          _mm512_exp_ps(r)
#elif defined(_LANE_256)
# define lane_load(l)         _mm256_load_ps(l)
# define lane_store(l,r)      _mm256_store_ps(l,r)
# define lane_widen(r)         _mm256_set1_ps(r)
# define lane_mul(l,r)        _mm256_mul_ps(l,r)
# define lane_div(l,r)        _mm256_div_ps(l,r)
# define lane_add(l,r)        _mm256_add_ps(l,r)
# define lane_sub(l,r)        _mm256_sub_ps(l,r)
# define lane_muladd(l,r0,r1) _mm256_fmadd_ps(r0,r1,l)
# define lane_exp(r)          _mm256_exp_ps(r)
#elif defined(_LANE_128)
# define lane_load(l)         _mm_load_ps(l)
# define lane_store(l,r)      _mm_store_ps(l,r)
# define lane_widen(r)        _mm_set1_ps(r)
# define lane_mul(l,r)        _mm_mul_ps(l,r)
# define lane_div(l,r)        _mm_div_ps(l,r)
# define lane_add(l,r)        _mm_add_ps(l,r)
# define lane_sub(l,r)        _mm_sub_ps(l,r)
# define lane_muladd(l,r0,r1) _mm_fmadd_ps(r0,r1,l)
#else
# define lane_load(l)         (*(l))
# define lane_store(l,r)      (lane_load(l)=(r))
# define lane_widen(r)        (r)
# define lane_mul(l,r)        ((l)*(r))
# define lane_div(l,r)        ((l)/(r))
# define lane_add(l,r)        ((l)+(r))
# define lane_sub(l,r)        ((l)-(r))
# define lane_muladd(l,r0,r1) lane_add(l,lane_mul(r0,r1))
# define lane_exp(r)          expf(r)
#endif


// context:
// Structure for storing vector vector types of arbitrary length,
// the actual amount of memory allocated is a multiple of the lane size,
// and it is stored in the max member.
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
// context: 
// layer structure: 
// -     wei: which is the weight matrix,
// -     bia: which is the bias vector,
// -     act: which is the activated output,
// -     err: which is error term propagated backwards and also doubles as the bias gradient,
// - new_wei: which the new weight matrix
//
// Once a layer is "fed" backwards, we interpolate from the weight matrix (wei)
// to the new weight matrix (new_wei), and from the bias vector (bia) to the new 
// bias vector (err), thus learning.
//
typedef struct layer_t layer_t;
typedef struct layer_t
{ matrix_t wei;
  vector_t bia;
  vector_t act;
  vector_t err;
  matrix_t new_wei;
} layer_t;
// context:
// network structure:
// - lay_o: which is the final layer, the output layer at index 0,
// - lay_i: which is the first layer, the input layer at index ARRAY_SIZE-1,
// - layer: which is the intermediate layers,
// - alpha: which is the learning rate, the interpolation factor used for learning
typedef struct network_t network_t;
typedef struct network_t
{ layer_t   lay_o;
  layer_t   lay_i;
  layer_t   layer[0x10];
  float     alpha;
} network_t;

typedef struct trainer_t trainer_t;
typedef struct trainer_t
{ int             image_x;
  int             image_y;
  int             image_size;
  unsigned char * images;
  unsigned char * labels;
  int             length;
  matrix_t        target;
} trainer_t;

typedef struct sample_t sample_t;
typedef struct sample_t
{ vector_t        value;
  int             label;
  vector_t        target;
} sample_t;

ccfunc ccinle unsigned int xorshift32(unsigned int x);
ccfunc ccinle double xorshift_randreal64(double min, double max);
ccfunc ccinle float  xorshift_randreal32(float min, float max);

// context: 
// - The following vector operator functions expect properly aligned, (aliasing-allowed)
// and padded memory, this is to facilitate the use of SIMD, for that reason, the 
// underlying memory should be a multiple of _LANE_SIZE*_LOOP_UNROLL, no less than 
// _LANE_SIZE*_LOOP_UNROLL and properly aligned. The default allocator takes this into account.
//
// - The first parameter is the length of the vectors component wise, the underlying
// allocation size of the memory is irrelevant as it is assumed to be properly 
// allocated as aforementioned.
// - The second parameter is the destination operand, then the left operand and finally
// the right operand.
//
ccfunc ccinle void vec_mul(int, float *, float *, float *);
ccfunc ccinle void vec_div(int, float *, float *, float *);
ccfunc ccinle void vec_add(int, float *, float *, float *);
ccfunc ccinle void vec_sub(int, float *, float *, float *);
ccfunc ccinle void vec_dot(int, float *, float *, float *);
// context:
// - This functions takes a length, a destination operand, 
// a left and right operand and an alpha factor, the right operand is multiplied
// by alpha and added to the left operand, the result is stored in destination.
// - destination = left + right * alpha
ccfunc ccinle void vec_muladd(int, float *, float *, float *, float alpha);
// context:
// - This function takes a single destination vector operand and it sets its
// memory to zero.
ccfunc ccinle void vec_zro(int, float *);

ccfunc ccinle unsigned int xorshift32(unsigned int x)
{ x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

// todo!: this isn't proper, see https://prng.di.unimi.it ...
ccfunc ccinle float xorshift_randreal32(float min, float max)
{ ccglobal ccthread_local unsigned int state = 901823102;
  state = xorshift32(state);

  double v=(double)state/~0u;
  return (float)(min+(max-min)*v);
}

ccfunc ccinle void
vec_mul(int len, float *dst, float *lhs, float *rhs)
{
  for(int i=0; i<len; i+=_LANE_SIZE)
    lane_store(dst+i,lane_mul(lane_load(lhs+i),lane_load(rhs+i)));
}

ccfunc ccinle void
vec_div(int len, float *dst, float *lhs, float *rhs)
{
  for(int i=0; i<len; i+=_LANE_SIZE)
    lane_store(dst+i,lane_div(lane_load(lhs+i),lane_load(rhs+i)));
}

ccfunc ccinle void
vec_add(int len, float *dst, float *lhs, float *rhs)
{
  for(int i=0; i<len; i+=_LANE_SIZE)
    lane_store(dst+i,lane_add(lane_load(lhs+i),lane_load(rhs+i)));
}

ccfunc ccinle void
vec_sub(int len, float *dst, float *lhs, float *rhs)
{
  for(int i=0; i<len; i+=_LANE_SIZE)
    lane_store(dst+i,lane_sub(lane_load(lhs+i),lane_load(rhs+i)));
}

ccfunc ccinle void
vec_muladd(int len, float *dst, float *lhs, float *rhs, float alpha)
{
  for(int i=0; i<len; i+=_LANE_SIZE)
    lane_store(dst+i,lane_muladd(lane_load(lhs+i),lane_load(rhs+i),lane_widen(alpha)));
}

ccfunc ccinle void
vec_dot(int len, 
    float * dst, 
    float * lhs, 
    float * rhs)
{
  lane_t acc[_LOOP_UNROLL];

#if _LOOP_UNROLL >= 1
  acc[0]=lane_widen(0);
#if _LOOP_UNROLL >= 2
  acc[1]=lane_widen(0);
#if _LOOP_UNROLL >= 4
  acc[2]=lane_widen(0);
  acc[3]=lane_widen(0);
#endif//_LOOP_UNROLL >= 1
#endif//_LOOP_UNROLL >= 2
#endif//_LOOP_UNROLL >= 4

  for(int i=0; i<len; i += _LOOP_UNROLL*_LANE_SIZE)
  {
#if _LOOP_UNROLL >= 1
    __assume((uintptr_t)(lhs+i+_LANE_SIZE*0)%_LANE_SIZE == 0);
    acc[0]=lane_muladd(acc[0],
      lane_load(lhs+i+_LANE_SIZE*0),lane_load(rhs+i+_LANE_SIZE*0));
#if _LOOP_UNROLL >= 2
    __assume((uintptr_t)(lhs+i+_LANE_SIZE*1)%_LANE_SIZE == 0);
    acc[1]=lane_muladd(acc[1],
      lane_load(lhs+i+_LANE_SIZE*1),lane_load(rhs+i+_LANE_SIZE*1));
#if _LOOP_UNROLL >= 4
    __assume((uintptr_t)(lhs+i+_LANE_SIZE*2)%_LANE_SIZE == 0);
    acc[2]=lane_muladd(acc[2],
      lane_load(lhs+i+_LANE_SIZE*2),lane_load(rhs+i+_LANE_SIZE*2));
    __assume((uintptr_t)(lhs+i+_LANE_SIZE*3)%_LANE_SIZE == 0);
    acc[3]=lane_muladd(acc[3],
      lane_load(lhs+i+_LANE_SIZE*3),lane_load(rhs+i+_LANE_SIZE*3));
#endif//_LOOP_UNROLL >= 1
#endif//_LOOP_UNROLL >= 2
#endif//_LOOP_UNROLL >= 4
  }

#if _LOOP_UNROLL >= 1
  acc[0];
#if _LOOP_UNROLL >= 2
  acc[0]=lane_add(acc[0],acc[1]);
#if _LOOP_UNROLL >= 4
  acc[2]=lane_add(acc[2],acc[3]);
  acc[0]=lane_add(acc[0],acc[2]);
#endif//_LOOP_UNROLL >= 1
#endif//_LOOP_UNROLL >= 2
#endif//_LOOP_UNROLL >= 4

  float v=0.f;
  for(int i=0;i<_LANE_SIZE;i+=1)
    v+=((float*)acc)[i];
  *dst=v;
}

ccfunc ccinle void
vec_zro(int len, float *dst)
{
  lane_t set=lane_widen(0.f);
  for(int i=0; i<len; i+=_LANE_SIZE)
    lane_store(dst+i,set);
}

// todo: this could be made purely intrinsic
ccfunc ccinle void
vec_rnd(int len, float *dst)
{
  for(int i=0; i<len; ++i)
    dst[i]=xorshift_randreal32(-1.f,+1.f);
}

ccfunc ccinle int
vector_allocation_size(int len)
{
  int z = _LOOP_UNROLL*_LANE_SIZE;

  return (len + z - 1) / z * z;
}

#ifndef del_vec
# define del_vec(vec) _aligned_free(vec.mem);
#endif

ccfunc ccinle vector_t
vector(int len)
{ int max=vector_allocation_size(len);
  void *mem=_aligned_malloc(sizeof(float)*max,0x20);
  vec_zro(max,(float*)mem);

  vector_t v;
  v.len=len;
  v.max=max;
  v.mem=(float *)mem;
  return v;
}

// For a col matrix, the number of vectors has to the be the number of columns
// and the vector size has to be the number of rows.
#ifndef new_col_mat
# define new_col_mat(col,row) matrix(col,row,col,row)
#endif
// For a row matrix, the number of vectors has to the be the number of rows
// and the vector size has to be the number of columns.
#ifndef new_row_mat
# define new_row_mat(col,row) matrix(col,row,row,col)
#endif
#ifndef matrix_clone
# define matrix_clone(mat) matrix(mat.col,mat.row,mat.min,mat.vec_min)
#endif
// note: when I say new row matrix, I mean that the number of vectors is the same as the number of rows.
ccfunc matrix_t
matrix(int col, int row, int min, int vec)
{
  matrix_t m;
  m.col     = col;
  m.row     = row;
  m.min     = min;
  m.max     = min;
  m.vec_min = vec;
  m.vec_max = vector_allocation_size(vec);

  m.mem = (float *) _aligned_malloc(sizeof(float) * m.vec_max * min, 0x20);
  memset(m.mem, 0, sizeof(float) * m.vec_max * m.min);
  return m;
}

ccfunc ccinle vector_t
matrix_vector(matrix_t matrix, int index)
{ vector_t v;
  v.mem = matrix.mem + matrix.vec_max * index;
  v.len = matrix.vec_min;
  v.max = matrix.vec_max;
  return v;
}

// note: you can also think of this as, how many inputs, how many outputs, in that order...
ccfunc layer_t
create_layer(int col_len, int row_len)
{ layer_t n;
  n.wei     = new_row_mat(col_len,row_len);
  n.act     = vector(row_len);
  n.bia     = vector(row_len);
  n.err     = vector(row_len);
  // todo: this is what I'm using to cache the new set of weights and later interpolate ...
  n.new_wei = matrix_clone(n.wei);

  vec_rnd(n.wei.min*n.wei.vec_max,n.wei.mem);
  vec_rnd(n.bia.len,n.bia.mem);
  return n;
}

// context: feeds the layer with input vector 'x',
// returns the activated output of the layer.
ccfunc ccinle vector_t
layer_feed(layer_t *layer, vector_t x)
{
  // context: the weight matrix
  matrix_t w=layer->wei;
  // context: the bias vector
  vector_t b=layer->bia;
  // context: the activated output of this layer
  vector_t a=layer->act;

  // context: ensure this layer is valid and can 
  // handle input vector 'x'
  ccassert(x.len==w.col);
  ccassert(b.len==w.row);
  ccassert(a.len==w.row);

  // context: compute the output of this layer, the product of our weight 
  // matrix and the input vector x.
  int i;
  for(i=0;i<w.row;++i)
    vec_dot(x.len,a.mem+i,w.mem+w.vec_max*i,x.mem);
  // context: add the bias vector to the 'un-activated' output
  vec_add(a.len,a.mem,a.mem,b.mem);

  // todo: have a callback here for custom activation functions
  lane_t n=lane_widen(1);
  lane_t z=lane_widen(0);

  // context: compute the activated output, we're using sigmoid
  // where sigmoid is defined as sigmoid(o) := (1.0/(1.0+exp(-o)))
  for(i=0; i<a.len; i+=_LANE_SIZE)
    lane_store(a.mem+i,
      lane_div(n,lane_add(n,lane_exp(lane_sub(z,lane_load(a.mem+i))))));
  return a;
}

ccfunc ccinle void
layer_update(layer_t *lay, float alpha)
{
  // note: interpolate towards the new set of weights and biases, note how we use
  // negative alpha... this is because we want to reduce the error, the cost function
  // tells use how far we are form the ideal result, this is our error...

  vector_t b=lay->bia;
  vec_muladd(b.len,b.mem,b.mem,lay->err.mem,-alpha);

  matrix_t w=lay->wei,wei_new=lay->new_wei;
  vec_muladd(w.min*w.vec_max,w.mem,w.mem,wei_new.mem,-alpha);
}

ccfunc ccinle void
network_init(network_t *net, int inp, int con, int out)
{ net->lay_i=create_layer(inp,con);
  net->lay_o=create_layer(con,out);
}

ccfunc ccinle void
network_feed(network_t *net, vector_t inp)
{
  layer_feed(& net->lay_o,
    layer_feed(& net->lay_i, inp));
}

ccfunc ccinle void
network_reverse_feed(network_t * network, vector_t inp_v, vector_t tar_v)
{
  layer_t
    lay_o=network->lay_o,
    lay_i=network->lay_i;
  vector_t
    act_o=lay_o.act,
    err_o=lay_o.err;
  vector_t
    act_i=lay_i.act,
    err_i=lay_i.err;
  matrix_t
    wei_o     = lay_o.wei,
    wei_i     = lay_i.wei,
    new_wei_o = lay_o.new_wei,
    new_wei_i = lay_i.new_wei;

  ccassert(inp_v.len==wei_i.col);

  int row,col,i;
  float acc;

  lane_t n=lane_widen(1);
  lane_t z=lane_widen(0);

  //
  // ** Here's our cost function:
  //
  // `E := 1/2 * pow<2>(O[0:j] - Y[d:j])`
  //
  // ** Note that the cost function is only defined for the outputs of neurons
  // at layer `0`, the output layer.
  //
  // `E\\A[k:j] := E[d:j]\\O[k:j] * O[k:j]\\A[k:j]`
  //
  // ** Since `E` is a function of `O` and not `A`, we have to first find how much
  // did `O` affect `E`, then find out how much did `A` affect `O`, and the result
  // is how much did `A` affect `E`.
  //
  // ** We get this:
  //
  // `E[d:j]\\A[k:j] := (O[k:j] - Y[k:j]) * sig"(A[k:j])`
  //
  // ** And we call it the error, we use '$' to denote it.
  //
  // `$[k:j] := (O[k:j] - Y[k:j]) * sig"(A[k:j])`
  //
  // `for(j..r[k]):
  //    $[k:j] = (O[k:j] - Y[k:j]) * sig"(A[k:j])`
  //
  for(i=0; i<err_o.len; i+=_LANE_SIZE)
  { lane_t a=lane_load(act_o.mem+i);
    lane_t y=lane_load(tar_v.mem+i);
    lane_store(err_o.mem+i,lane_mul(lane_sub(a,y),lane_mul(lane_sub(n,a),a)));
  }
  // ** We keep following the derivation chain, now let's find out how our weights and biases
  // affected `A`.
  //
  // `A[k:j]\\B[k:j] := $[k:j] * 1` (we don't have to do anything for this one)
  //
  // `A[k:j]\\W[k:j,i] := $[k:j] * O[k+1:i]`
  //
  // `for(j..r[k]):`
  // ` for(i..r[k+1]):`
  // `  A[k:j]\\W[k:j,i] = $[k:j] * O[k+1:i]`
  //
  // This reads as, for each index `j` of `r[k]`, the number of nodes in layer `k`,
  // and for each index `i` of `r[k+1]`, the number of nodes in layer `k+1`, the partial
  // the change in `A[k:j]` due to `W[k:j,i]` is the product of `$[k:j]` and `O[k+1:i]`.
  //
  ccassert(wei_o.col==wei_i.row);
  //
  // note: the number of columns or inputs of layer `k` is the same as the number of
  // outputs in layer `k+1`, hence `r[k]` = wei_o.row and `r[k+1]` = wei_o.col ..
  //
  for(row=0;row<wei_o.row;++row)
    for(col=0;col<wei_o.col;++col)
      new_wei_o.mem[row*new_wei_o.vec_max+col]=err_o.mem[row]*act_i.mem[col];
  //
  //
  // note: for the rest of the layers, where `k` is not `0`, not the output layer.
  //
  //
  // note: here is the cost function again:
  //
  // `E := 1/2 * pow<2>(O[0:j] - Y[d:j])`
  //
  // note: we're not at layer `0` anymore, so first, we need to figure out how much
  // did our output affect the output of other neurons, and because one output of ours
  // affects every other neuron in the previous layer this gets a little bit more intricate..
  //
  // E\\A[k:j] := sum(l..r[k-1], $[k-1:l] * A[k-1:l]\\A[k:l])
  //
  // for(j..r[k]):
  //   err[k:j]=sum(l..r[k-1]): $[k-1:l] * W[k-1:l:j] * sig"(a[k:j])
  //
  //
  ccassert(wei_i.row==wei_o.col);

  // note: calculate sig"(a[k:j]) lane wide
  for(i=0; i<wei_i.row; i+=_LANE_SIZE)
  { lane_t a=lane_load(act_i.mem+i);
    lane_store(err_i.mem+i,lane_mul(lane_sub(n,a),a));
  }


  // todo: I can't fathom a transpose being faster here, we have to test it out..
  for(row=0;row<wei_i.row;++row)
  { acc=0;
    for(col=0;col<wei_o.row;++col)
      acc+=err_o.mem[col]*wei_o.mem[col*wei_o.vec_max+row];
    err_i.mem[row]*=acc;
  }

  // todo: speed!
  for(row=0;row<wei_i.row;++row)
    for(col=0;col<wei_i.col;++col)
      new_wei_i.mem[row*new_wei_i.vec_max+col]=err_i.mem[row]*inp_v.mem[col];
}

ccfunc ccinle void
network_train(network_t *network, vector_t i, vector_t t)
{
  network_feed(network,i);
  network_reverse_feed(network,i,t);

  // todo!: actually update all the layers..
  layer_update(&network->lay_i,.1f);
  layer_update(&network->lay_o,.1f);
}

ccfunc int
network_prediction(network_t *net)
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


ccfunc ccinle unsigned int
int_reverse(unsigned int i)
{ return (((i & 0xFF000000) >> 0x18) |
          ((i & 0x000000FF) << 0x18) |
          ((i & 0x00FF0000) >> 0x08) |
          ((i & 0x0000FF00) << 0x08));
}

ccfunc sample_t
load_sample(trainer_t *trainer, vector_t buffer, int index)
{
  float normalize=1.f/255.f;

  unsigned char *image=trainer->images+trainer->image_size*index;
  sample_t sample;
  sample.label=trainer->labels[index];
  sample.target=matrix_vector(trainer->target,sample.label);

  // todo: this could be made SIMD too!
  for(size_t n=0; n<trainer->image_size; ++n)
    buffer.mem[n]=normalize*image[n];

  return sample;
}

ccfunc int
load_trainer(trainer_t *trainer)
{
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
    return 0;
  }

  unsigned char *labels=(unsigned char*)ccpullfile(labels_file,0,0);
  unsigned char *images=(unsigned char*)ccpullfile(images_file,0,0);

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
    return 0;
  }

  // note: decode the files ...
  int total_labels=int_reverse(((int *)labels)[1]);
  int total_images=int_reverse(((int *)images)[1]);

  if(total_images!=total_labels)
  {
    cctracewar("image count %i differs from label count %i", total_images,total_labels);
  }

  // todo: is this correct? was the y first? check out the source!
  trainer->image_y=int_reverse(((int *)images)[2]);
  trainer->image_x=int_reverse(((int *)images)[3]);
  trainer->image_size=trainer->image_x*trainer->image_y;

  trainer->labels=(unsigned char *)labels+sizeof(int)*2;
  trainer->images=(unsigned char *)images+sizeof(int)*3;

  trainer->length=total_images<total_labels?total_images:total_labels;

  trainer->target=new_row_mat(10,10);
  for(int i=0;i<10;++i)
    matrix_vector(trainer->target,i).mem[i]=1;
  return 1;
}

ccfunc int
network_predict(network_t *network, vector_t x)
{
  network_feed(network,x);
  return network_prediction(network);
}

void print_thisline(const char *s);
void print_nextline();

// note: sample usage...
ccfunc int
load_trained_network(network_t *network)
{
  trainer_t trainer;
  if(!load_trainer(&trainer))
    return 0;

  network_init(network,trainer.image_size,16,10);

  vector_t buffer=vector(trainer.image_size);

  int test_samples=10000;
  int sample_index;
  for(sample_index=0;sample_index<trainer.length-test_samples;++sample_index)
  { sample_t sample=load_sample(&trainer,buffer,sample_index);
    network_train(network,buffer,sample.target);
  }

  int correct_samples;
  for(correct_samples=0;sample_index<trainer.length;++sample_index)
  { sample_t sample=load_sample(&trainer,buffer,sample_index);
    network_feed(network,buffer);

    int p=network_prediction(network);
    int is_correct=sample.label==p;
    correct_samples+=is_correct;


    float accuracy=100.f*(float)correct_samples/test_samples;
    print_thisline(ccformat("accuracy: %.2f%%%%",accuracy));
  }
  print_nextline();

  float accuracy=100.f*(float)correct_samples/test_samples;
  cctracelog("test accuracy: %.2f%%%%",accuracy);

  return 1;
}



// Note: these you have to implement for other operating system...
#ifdef _WIN32
void print_thisline(const char *s)
{
  HANDLE h=GetStdHandle(STD_OUTPUT_HANDLE);

  CONSOLE_SCREEN_BUFFER_INFO i;
  GetConsoleScreenBufferInfo(h,&i);

  DWORD w;
  WriteConsoleOutputCharacter(h,s,(int)strlen(s),i.dwCursorPosition,&w);
}

void print_nextline()
{
  HANDLE h=GetStdHandle(STD_OUTPUT_HANDLE);

  CONSOLE_SCREEN_BUFFER_INFO i;
  GetConsoleScreenBufferInfo(h,&i);

  i.dwCursorPosition.Y++;
  SetConsoleCursorPosition(h,i.dwCursorPosition);
}
#else
void print_thisline(const char *s)
{
  printf(s);
  print_nextline();
}

void print_nextline()
{
  printf("\n");
}
#endif


#endif