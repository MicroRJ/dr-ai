#ifndef NUM_AI_HEADER_FILE
#define NUM_AI_HEADER_FILE


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


#define PI_APPROX (22.0/7.0)

#ifndef arr_len
# define arr_len(arr) ((int)((arr) ? (sizeof(arr)/sizeof((arr)[0])) : -1))
#endif
#ifndef sigmoid_approx
# define sigmoid_approx(o) (1.0/1.0+pow(PI_APPROX,o))
#endif

#ifndef activation_function
# define activation_function sigmoid
#endif
#ifndef activation_function
# define activation_function_prime sigmoid_prime
#endif
// TODO(RJ)!

#define ai_vec_func(return_type) static return_type

ai_vec_func(ai_float) squaref(ai_float val) { return val*val; }
ai_vec_func(ai_float) cube(ai_float val) { return val*val*val; }

typedef struct ai_lay
{ ai_mat wei;
  ai_vec bia;

  union
  { ai_vec new_bia;
    ai_vec err;
  };

  ai_vec tmp;

  ai_mat new_wei;
  ai_vec out;
  ai_vec act;
} ai_lay;
/**
 * For every network we're going to have at least two layers,
 * I and and O.
 *
 * The I handles external input.
 * The O handles internal output.
 *
 * The user must specify how many inputs the I layer takes,
 * how many outputs I layer produces, and how many outputs
 * the O layer produces.
 **/
typedef struct ai_net
{ ai_lay   lay_o;
  ai_lay   lay_i;
  ai_lay   layer[0x20];
  ai_float alpha;
} ai_net;


static ai_lay new_lay(ai_int col_len, ai_int row_len)
{ ai_lay n;

  n.wei = new_row_mat(col_len,row_len);
  n.act = new_vec(row_len);
  n.out = new_vec(row_len);
  n.bia = new_vec(row_len);

  n.err     = new_vec(row_len);
  n.tmp     = new_vec(row_len);
  n.new_wei = rpl_mat(n.wei);
  mat_rnd(n.wei);
  vec_rnd(n.bia);
  return n;
}

static void ai_net_ini(ai_net *net, ai_int inp, ai_int con, ai_int out)
{ net->lay_i = new_lay(inp, con);
  net->lay_o = new_lay(con, out);
}





static unsigned int __forceinline int_flip(unsigned i)
{ return (((i & 0xFF000000) >> 0x18) |
          ((i & 0x000000FF) << 0x18) |
          ((i & 0x00FF0000) >> 0x08) |
          ((i & 0x0000FF00) << 0x08));
}

static void *ai_read_file_data(unsigned int *file_read_result, const char *file_name)
{
  HANDLE file_handle =
    CreateFileA(file_name, GENERIC_READ, FILE_SHARE_WRITE|FILE_SHARE_READ, 0x00, OPEN_EXISTING, 0x00, 0x00);

  void *file_data =
    NULL;
  if(INVALID_HANDLE_VALUE != file_handle)
  { DWORD
      file_size    = 0,
      file_size_hi = 0,
      file_read    = 0;
    file_size = GetFileSize(file_handle, & file_size_hi);
    file_data = VirtualAlloc(NULL, file_size, MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);
    if(! ReadFile(file_handle, file_data, file_size, &file_read, 0x00))
    { UnloadFileData(file_data);
      file_data = NULL;
    }
    if(file_read_result) *file_read_result = file_read;
    CloseHandle(file_handle);
  }
  return file_data;
}

#endif