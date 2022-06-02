#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "utils.h"


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


FLOAT sum_weights(size_t size1, size_t size2, FLOAT *indweights1, FLOAT *indweights2, FLOAT *bitweights1, FLOAT *bitweights2, int n_bitwise_weights, int noffset, FLOAT default_value)
{
  FLOAT sumw = 0.;
  #pragma omp parallel for reduction(+:sumw)
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
