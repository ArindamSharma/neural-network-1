if g++ $1.cpp -o $1.out ; then
    ./$1.out
    rm $1.out
fi