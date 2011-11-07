#ifndef OPENCLFFTALGORITHM_H
#define OPENCLFFTALGORITHM_H

#include "uOpenCL.h"
#include "FFT.h"

template <class D> class OpenCLFFTAlgorithm : public FFT<D>, public OpenCLAlgorithm {
protected:
    void computeParameters(std::vector<int>& A, std::vector<int> B, std::vector<int>& N) {
        int b = 1;
        for (int q = 1; q <= B.size(); ++q) { A.push_back(b); b *= B[q - 1]; N.push_back(b); }
    }

    streng generateLocalFFTKernel(Array* buffs[2], std::vector<int> AL, std::vector<int> BL, std::vector<int> NL, int LL) { std::stringstream result;
        for (int qL = 1; qL <= BL.size(); ++qL) {
            Array& source = *buffs[(qL ^ 1) & 1];
            Array& target = *buffs[qL & 1];

            for (int gL = 0; gL < AL[qL - 1]; ++gL) { result << "{";
                for (int zL = 0; zL < (LL/NL[qL - 1]); ++zL) { result << "{";
                    for (int sL = 0; sL < BL[qL - 1]; ++sL) {
                        result << "const K s" << sL << " = "
                        << KomplexConstMultiplication(KomplexUnit(-gL*sL, NL[qL - 1]), source[(LL/NL[qL - 1])*(BL[qL - 1] * gL + sL) + zL])() << ";";
                    }

                    for (int hL = 0; hL < BL[qL - 1]; ++hL) {
                        result << target[(LL/NL[qL - 1])*(gL + AL[qL - 1]*hL) + zL]() << " = ";
                        for (int sL = 0; sL < BL[qL - 1] - 1; ++sL) result << "add(";
                        for (int sL = 0; sL < BL[qL - 1]; ++sL) {
                            if (sL > 0) result << ", ";
                            result << KomplexConstMultiplication(KomplexUnit(-hL*sL, BL[qL - 1]), streng("s") + intToStr(sL))();
                            if (sL > 0) result << ")";
                        }
                        result << ";";
                    }
                    result << "}";
                }
                result << "}";
            }
        }

        return result.str();
    }

public:
    OpenCLFFTAlgorithm(int iSize, int iBatchCount) : FFT<D>(iSize, iBatchCount), OpenCLAlgorithm() { }
};

#endif // OPENCLFFTALGORITHM_H
