/*
 * The information in this file is
 * Copyright (C) 2010-2011, Sven De Smet <sven@cubiccarrot.com>
 * and is subject to the terms and conditions of the
 * GNU Lesser General Public License Version 2.1
 * The license text is available from
 * http://www.gnu.org/licenses/lgpl.html
 */

#ifndef CONTIGUOUSKERNELGENERATOR_H
#define CONTIGUOUSKERNELGENERATOR_H

typedef std::string streng;

class KomplexUnit {
private:
    int numerator, denominator;
    bool negative;
public:
    KomplexUnit(int iNumerator, int iDenominator, bool iNegative = true) : numerator(iNumerator), denominator(iDenominator), negative(iNegative) { }
    streng toFloatString() {
        //double alpha = (negative ? -1 : 1)*2*M_PI*(((double) numerator)/denominator);
        std::stringstream result;
//        result << streng("unit(") << numerator << ", " << denominator << ")";
        result << streng("komplex(") << cos((2*M_PI*numerator)/denominator) << ", " << sin((2*M_PI*numerator)/denominator) << ")";
        return result.str();
    }
};

class KernelGenerator {
public:
static streng generateKernel(int kernelIx, int Aq, std::vector<int> BL, int LG, int NG, bool includeCommonDefs, bool preTwiddle = false) {
    int kL = BL.size();
    int LL = 1;
    int phi = 32;
    for (int qL = 1; qL <= kL; ++qL) LL *= BL[qL - 1];
    std::vector<int> AL, NL;
    int b = 1;
    for (int qL = 1; qL <= kL; ++qL) {
        AL.push_back(b);
        b *= BL[qL - 1];
        NL.push_back(b);
    }
    int Bq = LL;
    std::stringstream defines; defines << "\
    typedef float2 T2;\n\
//    const int kL = " << kL << ";\n\
    const int phi = " << phi << ";\n\
    const int LL = " << LL << ";\n";
    std::stringstream komplex; komplex << "\n\
typedef float T;\n\
struct Komplex { T r, i; };\n\
typedef struct Komplex K;\n\
inline K komplex(T iR, T iI);\n\
inline K unit(int n, int d);\n\
inline K mul(const K a, const K b);\n\
inline K add(const K a, const K b);\n\
inline K komplex(T iR, T iI) { K k; k.r = iR; k.i = iI; return k; };\n\
inline K unit(int n, int d) { const float PI = " << M_PI << "; return komplex(cos((2*PI*n)/d), sin((2*PI*n)/d)); };\n\
inline K mul(const K a, const K b) { return komplex(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r); };\n\
inline K add(const K a, const K b) { return komplex(a.r + b.r, a.i + b.i); };\n";
    std::stringstream kernelhead; kernelhead << "__kernel void contiguousFFT_step" << kernelIx << "(__global K *in, __global K *out) {\n";
    std::stringstream result; result << "\
    int unitsIx = 0;\n";
    int unitCount = 0;
    for (int qL = 1; qL <= kL; ++qL) { unitCount += NL[qL - 1]; }
    result << "\
//    int v = get_local_id(0); \n\
//    int w = get_group_id(0); \n\
    //int globalID = get_global_id(0); \n\
    int j = get_global_id(0);\n\
    //phi*w + v;\n\
    const int butterflyCount = " << LG/LL << ";\n\
    int gG = j/" << LG/NG << ";\n\
    int zG = j % " << LG/NG << ";\n\
    K buff0[LL];\n\
    K buff1[LL];\n\n";

    defines << "    const int fracLG_NG = " << LG/NG << ";\n";
    result << "  if (j < butterflyCount) {\n";
    // Load data
    std::stringstream subArrayReader; subArrayReader << "\
    int readStartOffset = zG + fracLG_NG*"<< Bq << "*gG;\n\
    for (int sG = 0; sG < " << Bq << "; ++sG) {\n";
    if (!preTwiddle) { subArrayReader << "        buff0[sG] = in[readStartOffset + sG*fracLG_NG];\n"; }
    else { subArrayReader << "      buff0[sG] = mul(in[readStartOffset + sG*fracLG_NG], unit(-(sG*gG), "<< NG << "));\n"; }
    //else { subArrayReader << "   int r = (sG*gG) & 1;   buff0[sG] = /*in[readStartOffset + sG*fracLG_NG],*/ pretwiddles[r];\n"; }
    subArrayReader << "\
    }\n";
    result << subArrayReader.str();
/*    result << "\
    int writeStartOffset = fracLG_NG*gG + zG;\n\
    for (int h = 0; h < " << Bq << "; ++h) {\n";
        result << "        out[writeStartOffset + h*" << Aq << "*fracLG_NG] = buff0[h];\n";
        //komplex(" << buffIx[kL & 1] << "[h].r, " << buffIx[kL & 1] << "[h].i);\n
    result << "    }/*";*/

    // Local FFT
    streng buffIx[2] = { "buff0", "buff1" };
    for (int qL = 1; qL <= kL; ++qL) {
        streng source = buffIx[(qL ^ 1) & 1];
        streng target = buffIx[qL & 1];

        defines << "    const int AL" << qL << " = " << AL[qL - 1] << ";\n";
        defines << "    const int BL" << qL << " = " << BL[qL - 1] << ";\n";
        defines << "    const int fracLL_NL" << qL << " = " << (LL/NL[qL - 1]) << ";\n";

        std::stringstream subkernel;
        for (int gL = 0; gL < AL[qL - 1]; ++gL) { subkernel << " { ";
            for (int sL = 0; sL < BL[qL - 1]; ++sL) subkernel << "        K u" << sL << " = "<< KomplexUnit(-gL*sL, NL[qL - 1]).toFloatString() << ";\n";

            std::stringstream innerKernel;
        for (int sL = 0; sL < BL[qL - 1]; ++sL) {
            innerKernel << "\
            K s" << sL << " = ";
            /*if (sL > 0)*/ innerKernel << "mul(u" << sL << ", ";
            innerKernel << source << "[fracLL_NL" << qL << "*(BL" << qL << "*" << gL << " + " << sL << ") + zL]);\n";
        }
        for (int hL = 0; hL < BL[qL - 1]; ++hL) {
            innerKernel << "\
            " << target << "[fracLL_NL" << qL << "*(" << gL << " + AL" << qL << "*" << hL << ") + zL] = ";
            for (int sL = 0; sL < BL[qL - 1] - 1; ++sL) innerKernel << "add(";
            for (int sL = 0; sL < BL[qL - 1]; ++sL) { if (sL > 0) innerKernel << ", "; innerKernel << "mul(s" << sL << ", " << KomplexUnit(-hL*sL, BL[qL - 1]).toFloatString() << ")"; if (sL > 0) innerKernel << ");\n"; }
        }

        subkernel << "\
        for (int zL = 0; zL < fracLL_NL" << qL << "; ++zL) {\n"
            << innerKernel.str() << "\n\
        }\n";
        subkernel << " } ";
        }
//        subkernel << "}";
        result << subkernel.str();
    }

    // Write results
    std::stringstream subArrayWriter; subArrayWriter << "\n\
    int writeStartOffset = fracLG_NG*gG + zG;\n\
    for (int h = 0; h < " << Bq << "; ++h) {\n";
        if (false) { subArrayWriter << "        out[writeStartOffset + h*" << Aq << "*fracLG_NG] = pretwiddles[h];;\n//" << buffIx[kL & 1] << "[h];\n"; }
        else { subArrayWriter << "        out[writeStartOffset + h*" << Aq << "*fracLG_NG] = " << buffIx[kL & 1] << "[h];\n"; }
        //komplex(" << buffIx[kL & 1] << "[h].r, " << buffIx[kL & 1] << "[h].i);\n
    subArrayWriter << "    }\n";
    result << subArrayWriter.str();
    result << "  }\n";

    result << "}";

    std::stringstream src;
    if (includeCommonDefs) src << komplex.str();
    src << kernelhead.str() << defines.str() << "\n" << result.str();
    return src.str();
}
};

#endif // CONTIGUOUSKERNELGENERATOR_H
