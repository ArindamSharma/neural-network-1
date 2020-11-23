export OMP_THREAD_NUM=5
if g++ $1.cpp -fopenmp -o $1.out ; then
    ./$1.out
    rm $1.out
fi