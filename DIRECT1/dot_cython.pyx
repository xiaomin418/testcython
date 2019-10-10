# dot_cython.pyx
import numpy as np
cimport numpy as np
from stdlib cimport *
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float32_t, ndim=2] _naive_dot(np.ndarray[np.float32_t, ndim=2] a, np.ndarray[np.float32_t, ndim=2] b):
    cdef np.ndarray[np.float32_t, ndim=2] c
    cdef int n, p, m
    cdef np.float32_t s
    if a.shape[1] != b.shape[0]:
        raise ValueError('shape not matched')
    n, p, m = a.shape[0], a.shape[1], b.shape[1]
    c = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            s = 0
            for k in range(p):
                s += a[i, k] * b[k, j]
            c[i, j] = s
    return c

def naive_dot(a, b):
    return _naive_dot(a, b)

def my_min(int x):
      return x+1


cdef class MulticlassParamData:
   cdef:
      double *acc
      double *w
      int *lastUpd
   def __cinit__(self, int nclasses):
      cdef int i
      self.lastUpd = <int *>malloc(nclasses*sizeof(int))
      self.acc     = <double *>malloc(nclasses*sizeof(double))
      self.w       = <double *>malloc(nclasses*sizeof(double))
      for i in range(nclasses):
         self.lastUpd[i]=0
         self.acc[i]=0
         self.w[i]=0

   def __dealloc__(self):
      free(self.lastUpd)
      free(self.acc)
      free(self.w)


cdef class MultitronParameters:
   cdef:
      int nclasses
      int now
      dict W

      double* scores # (re)used in calculating prediction

   def __cinit__(self, nclasses):
      self.scores = <double *>malloc(nclasses*sizeof(double))

   cpdef getW(self, clas):
      d={}
      cdef MulticlassParamData p
      for f,p in self.W.iteritems():
         d[f] = p.w[clas]
      return d

   def __init__(self, nclasses):
      self.nclasses = nclasses
      self.now = 0
      self.W = {}

   cdef _tick(self):
      self.now=self.now+1

   def tick(self): self._tick()

   cpdef scalar_multiply(self, double scalar):
      """
      note: DOES NOT support averaging
      """
      cdef MulticlassParamData p
      cdef int c
      for p in self.W.values():
         for c in range(self.nclasses):
            p.w[c]*=scalar

   cpdef add(self, list features, int clas, double amount):
      cdef MulticlassParamData p
      for f in features:
         try:
            p = self.W[f]
         except KeyError:
            p = MulticlassParamData(self.nclasses)
            self.W[f] = p

         p.acc[clas]+=(self.now-p.lastUpd[clas])*p.w[clas]
         p.w[clas]+=amount
         p.lastUpd[clas]=self.now

   cpdef add_r(self, list features, int clas, double amount):
      """
      like "add", but with real values features: 
         each feature is a pair (f,v), where v is the value.
      """
      cdef MulticlassParamData p
      cdef double v
      cdef str f
      for f,v in features:
         try:
            p = self.W[f]
         except KeyError:
            p = MulticlassParamData(self.nclasses)
            self.W[f] = p

         p.acc[clas]+=(self.now-p.lastUpd[clas])*p.w[clas]
         p.w[clas]+=amount*v
         p.lastUpd[clas]=self.now

   cpdef set(self, list features, int clas, double amount):
      """
      like "add", but replaces instead of adding
      """
      cdef MulticlassParamData p
      cdef double v
      cdef str f
      for f in features:
         try:
            p = self.W[f]
         except KeyError:
            p = MulticlassParamData(self.nclasses)
            self.W[f] = p

         p.acc[clas]+=(self.now-p.lastUpd[clas])*p.w[clas]
         p.w[clas]+=amount
         p.lastUpd[clas]=self.now
   def plusone(self):
       return self.nclasses+1





















