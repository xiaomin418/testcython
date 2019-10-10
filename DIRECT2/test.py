import cython
import pyximport
pyximport.install()
from DIRECT1 import dot_cython
from dot_cython import MultitronParameters
a=MultitronParameters(4)
print("a.tick: ",a.plusone())
print("hello")
