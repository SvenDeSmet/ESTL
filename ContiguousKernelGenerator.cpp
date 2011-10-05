
#include "ContiguousKernelGenerator.h"
#include "stdio.h"

std::string intToStr(int i) { if (i == 0) return "0";
    char p[(int) (ceil(log(i)/log(10)) + 2)];
    sprintf(p, "%i", i);
    return p;
}

int mod(int a, int b) { return (a - b*(a/b)); }
