/* The information in this file is
 * Copyright (C) 2011, Sven De Smet <sven@cubiccarrot.com>
 * and is subject to the terms and conditions of the
 * GNU Lesser General Public License Version 2.1
 * The license text is available from
 * http://www.gnu.org/licenses/lgpl.html
 */

#ifndef FFT_OPENCL_LAC_H
#define FFT_OPENCL_LAC_H

#include <vector>
#include <sstream>
#include <string>
#include <exception>
#include <sys/types.h>

#include "uOpenCL.h"
#include "FFT.h"
#include "tests/Timer.h"
#include "SimpleMath.h"

#define DBG(a) a

template <class D> class FFT_OpenCL_LAC : public FFT<D>, public OpenCLAlgorithm {
public:
    typedef PlannarizedDataInterface<D> DataInterfaceType;
    //typedef SplitInterleavedDataInterface<D> DataInterfaceType;
private:
    ComplexArrayCL<D>* data[2];
    bool forward;
    std::vector<cl::Kernel*> kernels;
    std::vector<cl::Event> kernelEvents;
    CLProgram* program;
    std::vector<streng> src;
    DataInterfaceType* arrayDataInterface;
public:
    static streng getName() { return "Shared memory, contiguous OpenCL FFT"; }

    std::vector<int> AGs, BGs, NGs;

    static int getRequiredMemory(int n) { return 3*sizeof(Complex<D>)*n; }
    static int getMaxBatchSize(int n) { return getGlobalMemory()/getRequiredMemory(n); }

    FFT_OpenCL_LAC(int iSize, bool iForward = true, int iBatchCount = 1) : FFT<D>(iSize, iBatchCount), OpenCLAlgorithm(), forward(iForward), program(NULL) {
        this->dataInterface = arrayDataInterface = new DataInterfaceType(this->batchCount*this->size);

        if (getMaxBatchSize(this->size) < this->batchCount) { printf("Insufficient global memory"); } // throw exception

        data[0] = new ComplexArrayCL<float>(*context, arrayDataInterface->in);
        data[1] = new ComplexArrayCL<float>(*context, arrayDataInterface->out);

        if (this->size > 1) {
            int defaultBGl2 = getDefaultLocalityFactorL2(0);
            int log2Size = ceilog(this->size, 2);
            int stepsG = (log2Size + (defaultBGl2 - 1))/defaultBGl2;

            BGs = std::vector<int>(log2Size/defaultBGl2, 1 << defaultBGl2);
            if (log2Size != ceilint(log2Size, defaultBGl2)) BGs.push_back(1 << (log2Size % defaultBGl2));
            computeParameters(AGs, BGs, NGs);
            for (int qG = 1; qG <= stepsG; ++qG) src.push_back(generateKernel(qG - 1));

            program = new CLProgram(*context, src, devicesToUse);
            for (int qG = 1; qG <= stepsG; ++qG) kernels.push_back(program->getKernel(streng("contiguousFFT_step") + intToStr(qG)));
        }

        this->kernelTimers.resize(kernels.size()); // printf("Initialization complete"); fflush(stdout);
    }

    double getTime(int first, bool firstStart, int last, bool lastStart) {
        cl_ulong start, end;
        clGetEventProfilingInfo(kernelEvents[last](), lastStart ? CL_PROFILING_COMMAND_START : CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        clGetEventProfilingInfo(kernelEvents[first](), firstStart ? CL_PROFILING_COMMAND_START : CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &start, NULL);
        return (end - start) * 1.0e-9f;
    }

    int getDefaultLocalityFactorL2(int level) { int factors[3] = { 3, 1, 1 }; return factors[level]; }

    // OpenCLAlgorithm
    virtual double getTotalComputationFlops(int kernel) { return (5.0*this->size)*(log(BGs[kernel])/log(2)); }
    virtual streng getKernelInfo(int kernel) { return intToStr(ceilog(BGs[kernel], 2)); }

    void printGPUDebugData(int qG) {
        data[(qG & 1) ^ 1]->enqueueReadArray(*commandQueue, *arrayDataInterface->in);
        printf("\n%i qG = %i", (qG & 1) ^ 1, qG); for (int q = 0; q < this->size; ++q) arrayDataInterface->in->getElement(q).print((q & 7) == 7);
    }

    int getSwarmSize(int qG) { return 1; } //mAx(1, mIn(16, mIn(this->size/NGs[qG - 1], getButterflyCount(qG)))); }
    int getButterflyCount(int qG) { return this->size/BGs[qG - 1]; }
    int getSwarmCount(int qG) { return getButterflyCount(qG)/getSwarmSize(qG); }
    int getThreadsPerWarpL2() { return 1; }
    int getThreadsPerWarp() { return 1 << getThreadsPerWarpL2(); }
    int getActualThreadsPerWarpL2(int qG) { return 0*mIn(getThreadsPerWarpL2(), ceilog(getSwarmCount(qG)/getWarpsPerKernel(qG), 2)); }
    int getThreadsPerBlock(int qG) { return getWarpsPerKernel(qG) << getActualThreadsPerWarpL2(qG); }
    int getWarpsPerKernel(int qG) { return mIn(4, mAx(1, 1 << mAx(0, (ceilog(BGs[qG - 1], 2) - getDefaultLocalityFactorL2(1))))); }
    int getWorkItemsPerKernel(int qG) { return getSwarmCount(qG)*getWarpsPerKernel(qG); }

    virtual void setKernelParameters(int kernelIx, cl::Kernel& kernel) { int q_G = kernelIx; int qG = q_G + 1;
        kernel.setArg<cl_mem>(0, data[(qG ^ 1) & 1]->getData());
        kernel.setArg<cl_mem>(1, data[(qG ^ 0) & 1]->getData());
    }

    virtual void execute() { //printf("Starting execution with %i kernels", (int) kernels.size()); fflush(stdout);
        if (this->size == 1) { arrayDataInterface->out->setElement(0, arrayDataInterface->in->getElement(0)); }
        else {
            for (int q = 0; q < this->size; ++q) arrayDataInterface->out->setElement(q, 888); data[1]->enqueueWriteArray(*commandQueue, *arrayDataInterface->out);
            data[0]->enqueueWriteArray(*commandQueue, *arrayDataInterface->in);

            kernelEvents.clear();
            for (int qG = 1; qG <= kernels.size(); ++qG) { cl::Kernel* kernel = kernels[qG - 1]; //printf("kernel q = %i", qG);
                size_t workGroupSize;
                kernel->getWorkGroupInfo<size_t>(devicesToUse[0], CL_KERNEL_WORK_GROUP_SIZE, &workGroupSize); // printf("[workGroupSize = %i]", workGroupSize);
                int workItems = getWorkItemsPerKernel(qG);
                int localSize = mIn(mIn(workItems, workGroupSize), getThreadsPerBlock(qG));

  //              printf("<<%i - %i>>", (ceilint(workItems, localSize)), (localSize));
                cl::KernelFunctor func = kernel->bind(*commandQueue, cl::NDRange(ceilint(workItems, localSize)), cl::NDRange(localSize));
                setKernelParameters(qG - 1, *kernel);
printGPUDebugData(qG);
                try { kernelEvents.push_back(func()); } // enqueue kernel + retain kernel event
                catch (cl::Error e) { printf("CL Exception"); CLException cle = CLException(e); cle.handle(); fflush(stdout); }
                catch (...) { printf("Unknown exception"); fflush(stdout); }
            } // printf("Computation ends..."); fflush(stdout);
printGPUDebugData(kernels.size() + 1);

            data[kernels.size() & 1]->enqueueReadArray(*commandQueue, *arrayDataInterface->out);

            kernelEvents[kernels.size() - 1].wait();
            if (timerComputation) timerComputation->addRun(getTime(0, true, kernels.size() - 1, false));
            for (int qG = 1; qG <= kernels.size(); ++qG) kernelTimers[qG - 1].addRun(getTime(qG - 1, true, qG - 1, false));
        }
    }

    virtual ~FFT_OpenCL_LAC() {
        for (int d = 0; d < 2; ++d) delete data[d];
        devicesToUse.clear();
        kernelEvents.clear();
        for (int k = 0; k < (int) kernels.size(); ++k) delete kernels[k];
        delete program;
    }

    void computeParameters(std::vector<int>& A, std::vector<int> B, std::vector<int>& N) {
        int b = 1;
        for (int q = 1; q <= B.size(); ++q) { A.push_back(b); b *= B[q - 1]; N.push_back(b); }
    }

/*
0 qG = 1(1.000000, 0.000000)(0.707107, 0.000000)(0.000000, 0.000000)(-0.707107, 0.000000)(-1.000000, 0.000000)(-0.707107, 0.000000)(-0.000000, 0.000000)(0.707107, 0.000000)
1 qG = 2(0.000000, 0.000000)(0.000000, 0.000000)(-0.000000, 0.000000)(0.000000, 0.000000)(2.000000, 0.000000)(1.414214, 0.000000)(0.000000, 0.000000)(-1.414214, 0.000000)
1 qG = 2(-0.000000, 0.000000)(0.000000, 0.000000)(2.000000, -0.000000)(1.414215, 1.414214)(0.000000, 0.000000)(0.000000, 0.000000)(2.000000, 0.000000)(1.414212, -1.414214)
0 qG = 3(-0.000000, 0.000000)(4.000001, -0.000001)(0.000000, 0.000000)(0.000001, 0.000004)(-0.000000, 0.000000)(-0.000001, 0.000001)(0.000000, 0.000000)(3.999999, -0.000004)
*/

virtual streng generateKernel(int q_G) { int qG = q_G + 1;
 //       printf("Generate Kernel (%i, %i): [%i] BM = ", qG, LG, BGs[q_G]);
        int LG = this->size;
        int plannarMask = (1 << GlobalPlannarLevel) - 1;
        int defaultBMl2 = getDefaultLocalityFactorL2(1); int defaultBM = 1 << defaultBMl2;
        int BGl2 = ceilog(BGs[q_G], 2);
        std::vector<int> BM = std::vector<int>(BGl2/defaultBMl2, defaultBM);
        if (BGl2 != ceilint(BGl2, defaultBMl2)) BM.push_back(1 << (BGl2 % defaultBMl2));
        int kM = BM.size();
        for (int q = 0; q < kM; ++q) printf("{%i}", BM[q]);
        std::vector<int> AM, NM;
        computeParameters(AM, BM, NM);
        //int swarmStrideLevel = 5;

        int phiL2 = getThreadsPerWarpL2();
        int sharedBufferLengthL = 2;
        int sharedBufferLengthM = getWarpsPerKernel(qG)*sharedBufferLengthL;
        int sharedDataSize = sharedBufferLengthM << getActualThreadsPerWarpL2(qG);

        std::stringstream result;
        if (qG == 1) result << KomplexMath::getDeclarations();
        result << "__kernel void contiguousFFT_step" << qG << "(__global float *in, __global float *out) {\n";
        result << "__local float sharedData[" << sharedDataSize << "];";
        result << "int jG = get_global_id(0);\n"; // = phi*w + v \n
          /*  int swarmIxOffset = globid >> " << swarmStrideLevel << ";\n\
            int subSwarmIx = globid & " << ((1 << swarmStrideLevel) - 1) << ";\n\
            for (int swarmIx = 0; swarmIx < " << swarmSize << "; ++swarmIx) {\n\
            int j = subSwarmIx + ((swarmIxOffset*" << swarmSize << " + swarmIx) << " << swarmStrideLevel << ");\n";*/
        result << "if (jG < " << (LG/BGs[q_G])*getWarpsPerKernel(qG) << ") {\n";
        printf("(wpk = %i)", getWarpsPerKernel(qG));
        result << "int jM = (jG >> " << getActualThreadsPerWarpL2(qG) << ") % " << getWarpsPerKernel(qG) << ";\n";
//        result << "out[jG] = jG; return;";
        printf("(actualThreadsPerWarp = %i)", 1 << getActualThreadsPerWarpL2(qG));
//        result << "out[jG] = jM; return;";
        result << "jG = ((jG >> " << getActualThreadsPerWarpL2(qG) << ") / " << getWarpsPerKernel(qG) << ") << " << getActualThreadsPerWarpL2(qG) << ";\n";
//        result << "out[jG] = jG; return;";

        result << "\
        int gG = jG/" << LG/NGs[q_G] << ";\n\
        int zG = jG - " << LG/NGs[q_G] << "*gG;\n";

        // M-Level FFT
        int maxBM = 0;
        for (int m = 0; m < (int) BM.size(); ++m) maxBM = mAx(BM[m], maxBM);
        Array buff0 = Array("K", maxBM, true, "buff0_");
        Array buff1 = Array("K", maxBM, true, "buff1_");
        Array* buff[2] = { &buff0, &buff1 };
        result << buff0.getDeclaration() << buff1.getDeclaration();

        int bX = 0;
        for (int qM = 1; qM <= BM.size(); ++qM) { int q_M = qM - 1; result << "{";
            result << "\
            int gM = jM/" << BGs[q_G]/NM[q_M] << ";\
            int zM = jM - " << BGs[q_G]/NM[q_M] << "*gM;";
            std::vector<int> AL, NL, BL = std::vector<int>(ceilog(BM[q_M], 2), 2);
            computeParameters(AL, BL, NL);
            int kL = BL.size();

            // G&M-Level Read (and pretwiddle)
            result << "int readStartOffsetM = zM + " << ((BGs[q_G]/NM[q_M]) * BM[q_M]) << "*gM;\n";
            if (qM == 1) {
                result << "int readStartOffsetG = zG + " << ((LG/NGs[q_G]) * BGs[q_G]) << "*gG;\n";
                for (int sM = 0; sM < BM[q_M]; ++sM) { result << "{"
                    << "int index = readStartOffsetG + (readStartOffsetM + " << sM*(BGs[q_G]/NM[q_M]) << ")*" << (LG/NGs[q_G]) << ";"
                    << "/*if (index < " << this->size << ")*/ { int ix = ((index >> " << GlobalPlannarLevel << ") << " << (GlobalPlannarLevel + 1) << ") | (index & " << plannarMask << ");"
                    << buff[bX]->getItem(sM)() << ".r = in[ix];"
                    << buff[bX]->getItem(sM)() << ".i = in[ix + " << (1 << GlobalPlannarLevel) << "];"
                    << "}}\n";
                }

                if (qG > 1) for (int sM = 0; sM < BM[q_M]; ++sM) { // Pretwiddle
                    result << buff[bX]->getItem(sM)()
                    << " = mul(" << buff[bX]->getItem(sM)() << ", unit(-(readStartOffsetM + " << sM*(BGs[q_G]/NM[q_M]) << ")*gG, "<< NGs[q_G] << ")" << ");\n";
                }
            } else { int q_MW = q_M - 1; result << "{"; // Local exchange
                result << "int coreIx = (jG & " << ((1 << getActualThreadsPerWarpL2(qG)) - 1) << ");";
                result << "for (int n = 0; n < " << sharedBufferLengthM << "; ++n) sharedData[((jM*" << sharedBufferLengthM << " + n) << " << getActualThreadsPerWarpL2(qG) << ") + coreIx] = 777;";
                result << "barrier(CLK_LOCAL_MEM_FENCE);\n";
                for (int l = 0; l < maxBM; l++) result << buff[bX ^ 1]->getItem(l)() << " = komplex(333.f, 44.f);";
                result << "\
                int gR = gM;\
                int zR = zM;\
                int gW = jM/" << BGs[q_G]/NM[q_MW] << ";\
                int zW = jM - " << BGs[q_G]/NM[q_MW] << "*gW;\
                int gR_div_AMW = (gR/" << AM[q_MW] << ");\
                int gR_mod_AMW = gR - " << AM[q_MW] << "*gR_div_AMW;\
                int zW_div_DMR = (zW/" << BGs[q_G]/NM[q_M] << ");\
                int zW_mod_DMR = zW - " << BGs[q_G]/NM[q_M] << "*zW_div_DMR;\
                ";
                if (BM[q_M] != BM[q_MW]) printf("Differing radices!!!");
                for (int syncStep = 0; syncStep < ceildiv(BM[q_M], sharedBufferLengthL); ++syncStep) {
                    printf("(syncSteps = %i, sharedDataSize = %i)", ceildiv(BM[q_M], sharedBufferLengthL), sharedDataSize);
/*  for (int sM = 0; sM < BM[q_M]; ++sM) { //
                    result << "out[2*(2*jG + jM) + " << sM << "] = " << buff[bX]->getItem(sM)() << ".r;";
                    result << "out[2*(2*jG + jM) + " << sM << " + " << (1 << GlobalPlannarLevel) << "] = " << buff[bX]->getItem(sM)() << ".i;";
  }
result << "return;";*/
//result << "if ((jM == 0) && (jG == 0)) { out[0] = " << buff[bX ^ 1]->getItem(1)() << ".r; } return;";
                    streng components[2] = { "r", "i" };
                    for (int c = 0; c < 2; ++c) { result << "{\n";
                        // Write
       //                 result << "if ((jM == 0) && (jG == 0)) out[0] = " << buff[bX]->getItem(1)() << ".r;"; result << "return;";
         //               result << "if ((jM == 0) && (jG == 0)) out[0] = (zW_div_DMR + zW_mod_DMR*" << (BM[q_M]) << ")*" << (sharedBufferLengthL << phiL2)
           //                                       << "+" << ((1 % sharedBufferLengthL) << phiL2) << " + coreIx; "; result << "return;";
  /*for (int sM = 0; sM < BM[q_M]; ++sM) { //
                    result << "out[2*(2*jG + jM) + " << sM << "] = " << buff[bX]->getItem(sM)() << ".r;";
                    result << "out[2*(2*jG + jM) + " << sM << " + " << (1 << GlobalPlannarLevel) << "] = " << buff[bX]->getItem(sM)() << ".i;";
  }
result << "return;";*/
                        for (int hW = sharedBufferLengthL*syncStep; hW < mIn(BM[q_M], sharedBufferLengthL*(syncStep + 1)); ++hW) {
                            result << "sharedData[(zW_div_DMR + zW_mod_DMR*" << (BM[q_M]) << ")*" << (sharedBufferLengthL << getActualThreadsPerWarpL2(qG))
                                                  << "+" << ((hW % sharedBufferLengthL) << getActualThreadsPerWarpL2(qG)) << " + coreIx] = "
                                << (*buff[bX])[hW]() << "." << components[c] << ";\n";
                        }
                        result << "barrier(CLK_LOCAL_MEM_FENCE);\n";
                        result << "for (int d = 0; d < " << this->size << "; ++d) out[d] = 555;";
    //result << "if ((jM == 0) && (jG == 0)) out[0] = sharedData[32]; return;";
//result << "if ((jM == 0)) for (int d = 0; d < " << sharedDataSize <<"; ++d) out[4*jG + d] = sharedData[d]; return;";
                        // Read
                        result << "if ((gR/" << sharedBufferLengthL << ") == " << syncStep << ") {\n";
                        for (int sR = 0; sR < BM[q_M]; ++sR) {
                            result << (*buff[bX ^ 1])[sR]() << "." << components[c]
                                << " = sharedData[gR_mod_AMW*" << (BM[q_M])*(sharedBufferLengthL << getActualThreadsPerWarpL2(qG)) << " + "<< sR*(sharedBufferLengthL << getActualThreadsPerWarpL2(qG))
                                                  << "+" << "((gR_div_AMW % "<< sharedBufferLengthL << ") << " << getActualThreadsPerWarpL2(qG) << ")" << " + coreIx];\n";
                        }
                        result << "}";
                        result << "barrier(CLK_LOCAL_MEM_FENCE);\n";
                        result << "}";
                    }
                }
                bX ^= 1;

// (0.000000, 0.000000)(-0.000000, 0.000000)(0.000000, 0.000000)(-0.000000, 0.000000)(2.000000, 0.000000)(0.000000, 0.000000)(2.000000, 0.000000)(0.000000, 0.000000)

  for (int sM = 0; sM < BM[q_M]; ++sM) { //
                    result << "out[2*(2*jG + jM) + " << sM << "] = " << buff[bX]->getItem(sM)() << ".r;";
                    result << "out[2*(2*jG + jM) + " << sM << " + " << (1 << GlobalPlannarLevel) << "] = " << buff[bX]->getItem(sM)() << ".i;";
  }
result << "return;";

                for (int sM = 0; sM < BM[q_M]; ++sM) { // Pretwiddle
                    result << buff[bX]->getItem(sM)()
                    << " = mul(" << buff[bX]->getItem(sM)() << ", unit(" << -sM << "*gM, "<< NM[q_M]<< ")" << ");\n";
                }

                result << "}";
            }

            // L-level FFT
            /*if (qM == 1)*/ for (int qL = 1; qL <= kL; ++qL) { int q_L = qL - 1;
                Array& source = *buff[bX ^ (qL & 1) ^ 1];
                Array& target = *buff[bX ^ (qL & 1)];

                for (int gL = 0; gL < AL[qL - 1]; ++gL) { result << "{";
                    for (int zL = 0; zL < (BM[q_M]/NL[q_L]); ++zL) { result << "{";
                        for (int sL = 0; sL < BL[qL - 1]; ++sL) {
                            result << "const K s" << sL << " = "
                            << KomplexConstMultiplication(KomplexUnit(-gL*sL, NL[qL - 1]), source.getItem((BM[q_M]/NL[qL - 1])*(BL[qL - 1] * gL + sL) + zL)).getRepresentation() << ";";
                        }

                        for (int hL = 0; hL < BL[qL - 1]; ++hL) {
                            result << target.getItem((BM[q_M]/NL[qL - 1])*(gL + AL[qL - 1]*hL) + zL).getRepresentation() << " = ";
                            for (int sL = 0; sL < BL[qL - 1] - 1; ++sL) result << "add(";
                            for (int sL = 0; sL < BL[qL - 1]; ++sL) {
                                if (sL > 0) result << ", ";
                                result << KomplexConstMultiplication(KomplexUnit(-hL*sL, BL[qL - 1]), streng("s") + intToStr(sL)).getRepresentation();
                                if (sL > 0) result << ")";
                            }
                            result << ";";
                        }
                        result << "}";
                    }
                    result << "}\n";
                }
            }

            // G&M-Level Write
//            result << "out[jM] = jM; return;";
            if (qM == 2) {
                result << "int writeStartOffsetG = zG + " <<       LG/NGs[q_G] << "*gG;\n";
                result << "int writeStartOffsetM = zM + " << BGs[q_G]/NM[q_M] << "*gM;\n";
                for (int hM = 0; hM < BM[q_M]; ++hM) { result << "{"
                    << "int index = writeStartOffsetG + (writeStartOffsetM + " << hM*AM[q_M]*(BGs[q_G]/NM[q_M]) << ")*" << AGs[q_G]*(LG/NGs[q_G]) << ";"
                    << "/*if (jM == 0)*/ if (index < " << this->size << ") { int ix = ((index >> " << GlobalPlannarLevel << ") << " << (GlobalPlannarLevel + 1) << ") | (index & " << plannarMask << ");"
                    << "out[ix] = " << (*buff[bX ^ (kL & 1)])[hM]() << ".r;"
                    << "out[ix + " << (1 << GlobalPlannarLevel) << "] = " << (*buff[bX ^ (kL & 1)])[hM]() << ".i;"
                    << "}}\n";
                }
//                 result << "out[jM] = jM;";
                result << "return;";
            }
            result << "}";
            bX = bX ^ (kL & 1);
        }

        result << "}}";
//        printf("#####################\n%s\n", result.str().c_str());
        return result.str();
    }
};

#endif // FFT_OPENCL_LAC_H
