/*
 * The information in this file is
 * Copyright (C) 2011, Sven De Smet <sven@cubiccarrot.com>
 * and is subject to the terms and conditions of the
 * GNU Lesser General Public License Version 2.1
 * The license text is available from
 * http://www.gnu.org/licenses/lgpl.html
 *
 * Disclaimer: IMPORTANT:
 *
 * The Software is provided on an "AS IS" basis.  Sven De Smet MAKES NO WARRANTIES,
 * EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION THE IMPLIED WARRANTIES OF
 * NON - INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE,
 * REGARDING THE SOFTWARE OR ITS USE AND OPERATION ALONE OR IN COMBINATION WITH YOUR PRODUCTS.
 *
 * IN NO EVENT SHALL Sven De Smet BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL OR
 * CONSEQUENTIAL DAMAGES ( INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION ) ARISING IN ANY WAY OUT OF THE USE, REPRODUCTION, MODIFICATION
 * AND / OR DISTRIBUTION OF THE SOFTWARE, HOWEVER CAUSED AND WHETHER
 * UNDER THEORY OF CONTRACT, TORT ( INCLUDING NEGLIGENCE ), STRICT LIABILITY OR
 * OTHERWISE, EVEN IF Sven De Smet HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CONTIGUOUSKERNELGENERATOR_H
#define CONTIGUOUSKERNELGENERATOR_H

#include <string>
#include <sstream>
#include "math.h"
#include <vector>
#include "Complex.h"

typedef std::string streng;

std::string intToStr(int i); /*{
    char p[(int) (ceil(log(i)/log(10)) + 2)];
    sprintf(p, "%i", i);
    return p;
}*/

class Expression {
public:
    virtual streng getRepresentation() = 0;
};

class Value : public Expression {
private:
    streng representation;
public:
    Value(streng iRepresentation) : representation(iRepresentation) { }
    Value() { }

    virtual streng getRepresentation() { return representation; }
};

class KomplexUnit : public Expression {
private:
    bool negative;
public:
    int numerator, denominator;

    KomplexUnit(int iNumerator, int iDenominator, bool iNegative = true) : numerator(iNumerator), denominator(iDenominator), negative(iNegative) { }
    streng toFloatString() {
        //double alpha = (negative ? -1 : 1)*2*M_PI*(((double) numerator)/denominator);
        std::stringstream result;
//        result << streng("unit(") << numerator << ", " << denominator << ")";
        result << streng("komplex(") << cos((2*M_PI*numerator)/denominator) << ", " << sin((2*M_PI*numerator)/denominator) << ")";
        return result.str();
    }

    virtual streng getRepresentation() { return toFloatString(); }
};

int mod(int a, int b); //{ return (a - b*(a/b)); }

class KomplexConstMultiplication : public Expression {
private:
    KomplexUnit a;
    Value b;
public:
    KomplexConstMultiplication(KomplexUnit iA, Value iB) : a(iA), b(iB) { }

    virtual streng getRepresentation() {
            streng bRep = b.getRepresentation();
            int num = mod(a.numerator, a.denominator);
            if (mod(4*num, a.denominator) == 0) { // 1 + 0 i --- 0 + 1 i --- -1 + 0 i --- 0 - 1 i
                if (num == 0) {
                    return bRep;
                } else if ((2*num == a.denominator) || (2*num == -a.denominator)) {
                    return streng("komplex(-") + bRep + streng(".r, -") + bRep + streng(".i)");
                } else if ((4*num == a.denominator) || (4*num == -3*a.denominator)) {
                    return streng("komplex(-") + bRep + streng(".i, ") + bRep + streng(".r)");
                } else {
                    return streng("komplex(") + bRep + streng(".i, -") + bRep + streng(".r)");
                }// else return streng("mul(") + a.getRepresentation() + streng(", ") + b.getRepresentation() + streng(")");
            } else return streng("mul(") + a.getRepresentation() + streng(", ") + b.getRepresentation() + streng(")");
    }
};

class Array {
private:
    streng name;
    streng type;
    int length;
    bool inRegisters;
public:
    Array(streng iType, int iLength, bool iInRegisters, streng iName) : type(iType), length(iLength), inRegisters(iInRegisters), name(iName) { }

    streng getDeclaration() {
        streng result;
        if (inRegisters) { result += type + streng(" ");
            for (int l = 0; l < length; ++l) {
                if (l > 0) result += ", ";
                result += name + intToStr(l);
            }
        } else {
            result = type + streng(" ") + name + streng("[") + intToStr(length) + streng("]");
        }
        return result + streng(";");
    }

    Value getItem(int ix) {
        if (inRegisters) { return name + intToStr(ix); }
        else { return name + streng("[") + intToStr(ix) + streng("]"); }
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
inline K unit(int n, int d) { const float frac_PI2_d = " << (float) (2*M_PI) << "f/d; return komplex(native_cos(frac_PI2_d*n), native_sin(frac_PI2_d*n)); };\n\
inline K mul(const K a, const K b) { return komplex(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r); };\n\
inline K add(const K a, const K b) { return komplex(a.r + b.r, a.i + b.i); };\n";
    std::stringstream kernelhead; kernelhead << "__kernel void contiguousFFT_step" << kernelIx << "(__global K *in, __global K *out) {\n";
    std::stringstream result;
    result << "\
//    int v = get_local_id(0); \n\
//    int w = get_group_id(0); \n\
    int j = get_global_id(0);\n\
    //phi*w + v;\n\
    const int butterflyCount = " << LG/LL << ";\n\
    int gG = j/" << LG/NG << ";\n\
    int zG = j % " << LG/NG << ";\n";

    Array buff0 = Array("K", LL, true, "buff0_");
    Array buff1 = Array("K", LL, true, "buff1_");

    result << buff0.getDeclaration() << "\n" << buff1.getDeclaration() << "\n";
    result << "  if (j < butterflyCount) {\n";
    // Load data
    std::stringstream subArrayReader; subArrayReader << "\
    __global float *inF = (__global float*) in;\n\
    __global float *outF = (__global float*) out;\n\
    int readStartOffset = zG + "<< (LG/NG) * Bq << "*gG;\n";
    int plannarMask = (1 << GlobalPlannarLevel) - 1;
    for (int sG = 0; sG < Bq; ++sG) { subArrayReader << "{";
        subArrayReader << "int index = readStartOffset + " << sG*(LG/NG) << ";";
        subArrayReader << "int ix = ((index >> " << GlobalPlannarLevel << ") << " << (GlobalPlannarLevel + 1) << ") | (index & " << plannarMask << ");";
//        subArrayReader << buff0.getItem(sG).getRepresentation() << " = in[readStartOffset + " << sG*(LG/NG) << "];\n";
        subArrayReader << buff0.getItem(sG).getRepresentation() << ".r = inF[ix];\n";
        subArrayReader << buff0.getItem(sG).getRepresentation() << ".i = inF[ix + " << (1 << GlobalPlannarLevel) << "];\n";
        subArrayReader << "}";
    }
    if (preTwiddle) for (int sG = 0; sG < Bq; ++sG) {
        subArrayReader << buff0.getItem(sG).getRepresentation()
        << " = mul(" << buff0.getItem(sG).getRepresentation() << ", unit(-(" << sG << "*gG), "<< NG << ")" << ");\n";
    }
    result << subArrayReader.str();

    // Local FFT
    Array* buffs[2] = { &buff0, &buff1 };
    for (int qL = 1; qL <= kL; ++qL) {
        Array& source = *buffs[(qL ^ 1) & 1];
        Array& target = *buffs[qL & 1];

        std::stringstream subkernel;
        for (int gL = 0; gL < AL[qL - 1]; ++gL) { subkernel << " { ";
            for (int zL = 0; zL < (LL/NL[qL - 1]); ++zL) { subkernel << " { ";
                std::stringstream innerKernel;
                for (int sL = 0; sL < BL[qL - 1]; ++sL) {
                    innerKernel << "\
                    const K s" << sL << " = ";
                        innerKernel << KomplexConstMultiplication(KomplexUnit(-gL*sL, NL[qL - 1]),
                                 source.getItem((LL/NL[qL - 1])*(BL[qL - 1] * gL + sL) + zL)).getRepresentation() << ";";
                }
                for (int hL = 0; hL < BL[qL - 1]; ++hL) {
                    innerKernel << target.getItem((LL/NL[qL - 1])*(gL + AL[qL - 1]*hL) + zL).getRepresentation() << " = ";
                    for (int sL = 0; sL < BL[qL - 1] - 1; ++sL) innerKernel << "add(";
                    for (int sL = 0; sL < BL[qL - 1]; ++sL) {
                        if (sL > 0) innerKernel << ", ";
                        innerKernel << KomplexConstMultiplication(KomplexUnit(-hL*sL, BL[qL - 1]), streng("s") + intToStr(sL)).getRepresentation();
                        if (sL > 0) innerKernel << ")";
                    }
                    innerKernel << ";\n";
                }

                subkernel << innerKernel.str();
                subkernel << " } ";
            }
            subkernel << " } ";
        }
        result << subkernel.str();
    }

    // Write results
    std::stringstream subArrayWriter; subArrayWriter << "\n\
    int writeStartOffset = " << LG/NG << "*gG + zG;\n";
    for (int h = 0; h < Bq; ++h) { subArrayWriter << "{";
        subArrayWriter << "int index = writeStartOffset + " << h*Aq*(LG/NG) << ";";
        subArrayWriter << "int ix = ((index >> " << GlobalPlannarLevel << ") << " << (GlobalPlannarLevel + 1) << ") | (index & " << plannarMask << ");";
        //subArrayWriter << "        out[writeStartOffset + " << h*Aq*(LG/NG) << "] = " << buffs[kL & 1]->getItem(h).getRepresentation() << ";\n";
        subArrayWriter << "        outF[ix] = " << buffs[kL & 1]->getItem(h).getRepresentation() << ".r;\n";
        subArrayWriter << "        outF[ix + " << (1 << GlobalPlannarLevel) << "] = " << buffs[kL & 1]->getItem(h).getRepresentation() << ".i;\n";
        subArrayWriter << "}";
    }
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
