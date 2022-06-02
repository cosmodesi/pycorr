#ifndef	_UTILS_H_
#define	_UTILS_H_

#ifdef FLOAT32
typedef float FLOAT;
typedef int INTEGER;
#define POPCOUNT(X) __builtin_popcount(X)
#else
typedef double FLOAT;
typedef long INTEGER;
#define POPCOUNT(X) __builtin_popcountll(X)
#endif //_FLOAT32

#endif
