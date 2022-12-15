#include <math.h>
#include <stdio.h>

FLOAT mkname(_sum_weights)(const size_t size1, const size_t size2, FLOAT *indweights1, FLOAT *indweights2,
                           FLOAT *bitweights1, FLOAT *bitweights2, const int n_bitwise_weights, const int noffset, const FLOAT default_value, const int nthreads)
{
  FLOAT sumw = 0.;
#if defined(_OPENMP)
  set_num_threads(nthreads);
  #pragma omp parallel for reduction(+:sumw)
#endif
  for (size_t i1=0; i1<size1; i1++) {
    for (size_t i2=0; i2<size2; i2++) {
      FLOAT weight = 1.;
      if (indweights1 != NULL) weight *= indweights1[i1] * indweights2[i2];
      if (bitweights1 != NULL) {
        int nbits = noffset;
        FLOAT *bw1 = bitweights1 + n_bitwise_weights * i1, *bw2 = bitweights2 + n_bitwise_weights * i2;
        for (int w=0; w<n_bitwise_weights; w++) {
            nbits += POPCOUNT(*((INTEGER *) &bw1[w]) & *((INTEGER *) &bw2[w]));
        }
        weight *= (nbits == 0) ? default_value : 1./nbits;
      }
      sumw += weight;
    }
  }
  return sumw;
}
