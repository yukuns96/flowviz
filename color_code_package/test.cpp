#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

int main() {
#ifdef _OPENMP
    std::cout << "OpenMP is enabled. _OPENMP value: " << _OPENMP << std::endl;
#else
    std::cout << "OpenMP is not enabled." << std::endl;
#endif
    return 0;
}
