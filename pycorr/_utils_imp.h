#if defined(_OPENMP)
#include <omp.h>
void set_num_threads(int num_threads)
{
  if (num_threads>0) omp_set_num_threads(num_threads);
}

int get_num_threads()
{
  //Calculate number of threads
  int num_threads=0;
#pragma omp parallel
  {
#pragma omp atomic
    num_threads++;
  }
  return num_threads;
}
#endif


#define mkname(a) a ## _ ## float
#define FLOAT float
#define INTEGER int
#define POPCOUNT(X) __builtin_popcount(X)
#include "_utils_generics.h"
#undef POPCOUNT
#undef FLOAT
#undef INTEGER
#undef mkname
#define mkname(a) a ## _ ## double
#define FLOAT double
#define INTEGER long
#define POPCOUNT(X) __builtin_popcountll(X)
#include "_utils_generics.h"
#undef POPCOUNT
#undef FLOAT
#undef INTEGER
#undef mkname
