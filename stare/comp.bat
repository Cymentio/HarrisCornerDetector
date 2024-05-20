nvcc -diag-suppress 550 main.cu test.cu -o main -Xcompiler /openmp
 
@del main.lib
@del main.exp
@del _CL*