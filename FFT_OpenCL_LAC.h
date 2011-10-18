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

#include "uOpenCL.h"
#include "FFT.h"
#include "LACKernelGenerator.h"
#include "tests/Timer.h"
#include <sys/types.h>

#define DBG(a) a

#include "SimpleMath.h"

template <class D> class FFT_OpenCL_LAC : public FFT<D>, public OpenCLAlgorithm {
public:
    typedef PlannarizedDataInterface<D> DataInterfaceType;
    //typedef SplitInterleavedDataInterface<D> DataInterfaceType;
private:
    ComplexArrayCL<D>* data[2];
    bool forward;
    std::vector<cl::Kernel*> kernels;
    std::vector<cl::Event> kernelEvents;
    cl::Program* program;
    cl::Program::Sources* source;
    std::vector<streng> src;
    DataInterfaceType* arrayDataInterface;
public:
    std::vector<int> AGs, BGs, NGs;

    FFT_OpenCL_LAC(int iSize, bool iForward = true, int iBatchCount = 1) : FFT<D>(iSize, iBatchCount), OpenCLAlgorithm(), forward(iForward) {
        this->dataInterface = arrayDataInterface = new DataInterfaceType(this->batchCount*this->size);

        int memReq = this->size*sizeof(clFFT_Complex)*this->batchCount;
        if(memReq >= CLDevice(devicesToUse[0]).globalMemorySize()) { printf("Insufficient global memory"); } // throw exception

        data[0] = new ComplexArrayCL<float>(*context, arrayDataInterface->in);
        data[1] = new ComplexArrayCL<float>(*context, arrayDataInterface->out);

        if (this->size > 1) {
            int defaultBGl2 = 2;
            int log2Size = ceilog(this->size, 2);
            int stepsG = (log2Size + (defaultBGl2 - 1))/defaultBGl2;

            source = new cl::Program::Sources();
            BGs = std::vector<int>(log2Size/defaultBGl2, 1 << defaultBGl2);
            if (log2Size != ceilint(log2Size, defaultBGl2)) BGs.push_back(1 << (log2Size % defaultBGl2));
            computeParameters(AGs, BGs, NGs);
            for (int qG = 1; qG <= stepsG; ++qG) { //printf("qG = %i", qG);
                //printf("Generate Kernel qG = %i, AG = %i, fracLG_NG - %i", qG, AG, this->size/NG);
                src.push_back(generateKernel(qG, this->size, getSwarmSize(qG)));
                //printf("%s\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", src[src.size() - 1].c_str());
                source->push_back(std::make_pair(src[src.size() - 1].c_str(), src[src.size() - 1].length()));
            }

            program = new cl::Program(*context, *source);
            try { program->build(devicesToUse, "-cl-mad-enable"); }
            catch (cl::Error cle) { printf("Error: %s", cle.what());
                if (cle.err() == CL_BUILD_PROGRAM_FAILURE) {
                    cl_build_status status;
                    program->getBuildInfo<cl_build_status>(devicesToUse[0], CL_PROGRAM_BUILD_STATUS, &status);
                    if (status != CL_SUCCESS) { try {
                        size_t ret_val_size;
                        clGetProgramBuildInfo((*program)(), devicesToUse[0](), CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
                        printf("size: %i", ret_val_size);
                        char* build_log = new char[ret_val_size + 1];
                        clGetProgramBuildInfo((*program)(), devicesToUse[0](), CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
                        build_log[ret_val_size] = '\0';
                        for (int s = 0; s < src.size(); ++s) printf("%s", src[s].c_str());
                        printf("Kernel build error:\n%s", build_log);
                        delete [] build_log;
                    } catch (cl::Error err) { printf("Kernel build error: unkown (Failed to retrieve build log)."); throw err; }
                    } else printf("Kernel build successful.");
                }
            } catch (std::exception e) { printf("%s", e.what()); }

            for (int qG = 1; qG <= stepsG; ++qG) {
                std::stringstream kernelName; kernelName << "contiguousFFT_step" << qG;
                cl_int err;
                kernels.push_back(new cl::Kernel(*program, kernelName.str().c_str(), &err)); xCLErr(err);
            }
        }

        this->kernelTimers.resize(kernels.size()); // printf("Initialization complete"); fflush(stdout);
    }

    double getTime(int first, bool firstStart, int last, bool lastStart) {
        cl_ulong start, end;
        clGetEventProfilingInfo(kernelEvents[last](), lastStart ? CL_PROFILING_COMMAND_START : CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        clGetEventProfilingInfo(kernelEvents[first](), firstStart ? CL_PROFILING_COMMAND_START : CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &start, NULL);
        return (end - start) * 1.0e-9f;
    }

    int getSwarmSize(int qG) { return 1; } //mAx(1, mIn(16, mIn(this->size/NGs[qG - 1], getButterflyCount(qG)))); }
    int getButterflyCount(int qG) { return this->size/BGs[qG - 1]; }
    int getSwarmCount(int qG) { return getButterflyCount(qG)/getSwarmSize(qG); }

    virtual double getTotalComputationFlops(int kernel) { return (5.0*this->size)*(log(BGs[kernel])/log(2)); }
    virtual streng getKernelInfo(int kernel) { return intToStr(ceilog(BGs[kernel], 2)); }

    void printGPUDebugData(int qG) {
        data[(qG & 1) ^ 1]->enqueueReadArray(*commandQueue, *arrayDataInterface->in);
        printf("\n%i qG = %i", (qG & 1) ^ 1, qG); for (int q = 0; q < this->size; ++q) arrayDataInterface->in->getElement(q).print();
     //   data[(qG & 1)]->enqueueReadArray(*commandQueue, *arrayDataInterface->in);
       // printf("1 qG = %i", qG); for (int q = 0; q < this->size; ++q) arrayDataInterface->in->getElement(q).print();
    }

    int getThreadsPerWarpL2() { return 0; }
    int getThreadsPerWarp() { return 1 << getThreadsPerWarpL2(); }
    int getWarpsPerKernel(int qG) { return mIn(1, mAx(1, (this->size/BGs[qG - 1])/getThreadsPerWarp())); }

    virtual void execute() { //printf("Starting execution with %i kernels", (int) kernels.size()); fflush(stdout);
        if (this->size == 1) { arrayDataInterface->out->setElement(0, arrayDataInterface->in->getElement(0)); }
        else { if (kernels.size() > 0) {
            for (int q = 0; q < this->size; ++q) arrayDataInterface->out->setElement(q, 888); data[1]->enqueueWriteArray(*commandQueue, *arrayDataInterface->out);
            data[0]->enqueueWriteArray(*commandQueue, *arrayDataInterface->in);

            kernelEvents.clear();
            for (int qG = 1; qG <= kernels.size(); ++qG) { cl::Kernel* kernel = kernels[qG - 1]; //printf("kernel q = %i", qG);
                size_t workGroupSize;
                kernel->getWorkGroupInfo<size_t>(devicesToUse[0], CL_KERNEL_WORK_GROUP_SIZE, &workGroupSize); // printf("[workGroupSize = %i]", workGroupSize);
                int localSize = mIn((this->size/BGs[qG - 1])*getWarpsPerKernel(qG), workGroupSize);

//                printf("<<%i - %i>>", (ceilint(getSwarmCount(qG), localSize)), (localSize));
           //     printf("wgs = %i --- size = %i -- s = %i -- localSize == %i -- butterFlyCount = %i", workGroupSize, this->size, s, localSize, butterflyCount);
                cl::KernelFunctor func = kernel->bind(*commandQueue, cl::NDRange(ceilint(getSwarmCount(qG)*getWarpsPerKernel(qG), localSize)), cl::NDRange(localSize));
                kernel->setArg<cl_mem>(0, data[(qG ^ 1) & 1]->getData());
                kernel->setArg<cl_mem>(1, data[(qG ^ 0) & 1]->getData());

//printGPUDebugData(qG);
                try { kernelEvents.push_back(func()); } // enqueue kernel + retain kernel event
                catch (cl::Error e) { printf("CL Exception"); CLException cle = CLException(e); cle.handle(); fflush(stdout); }
                catch (...) { printf("Unknown exception"); fflush(stdout); }
            } // printf("Computation ends..."); fflush(stdout);
//printGPUDebugData(kernels.size() + 1);

            data[kernels.size() & 1]->enqueueReadArray(*commandQueue, *arrayDataInterface->out);

            kernelEvents[kernels.size() - 1].wait();
            if (timerComputation) timerComputation->addRun(getTime(0, true, kernels.size() - 1, false));
            for (int qG = 1; qG <= kernels.size(); ++qG) kernelTimers[qG - 1].addRun(getTime(qG - 1, true, qG - 1, false));

//          printf("out"); for (int q = 0; q < this->size; ++q) this->out->getElement(q).print();
//          data[(kernels.size() & 1) ^ 1]->enqueueReadArray(*commandQueue, *this->in);
//          printf("in"); for (int q = 0; q < this->size; ++q) this->in->getElement(q).print();
        } }
    }

    virtual ~FFT_OpenCL_LAC() {
        for (int d = 0; d < 2; ++d) delete data[d];
        devicesToUse.clear();
        kernelEvents.clear();
        for (int k = 0; k < (int) kernels.size(); ++k) delete kernels[k];
        delete source;
        delete program;
    }

    void computeParameters(std::vector<int>& A, std::vector<int> B, std::vector<int>& N) {
        int b = 1;
        for (int q = 1; q <= B.size(); ++q) { A.push_back(b); b *= B[q - 1]; N.push_back(b); }
    }

    streng generateKernel(int qG, int LG, int swarmSize = 1) { int q_G = qG - 1;
 //       printf("Generate Kernel (%i, %i): [%i] BM = ", qG, LG, BGs[q_G]);
        int plannarMask = (1 << GlobalPlannarLevel) - 1;
        int defaultBMl2 = 2; int defaultBM = 1 << defaultBMl2;
        int BGl2 = ceilog(BGs[q_G], 2);
        std::vector<int> BM = std::vector<int>(BGl2/defaultBMl2, defaultBM);
        if (BGl2 != ceilint(BGl2, defaultBMl2)) BM.push_back(1 << (BGl2 % defaultBMl2));
        int kM = BM.size();
  //      for (int q = 0; q < kM; ++q) printf("%i*", BM[q]);
        std::vector<int> AM, NM;
        computeParameters(AM, BM, NM);
        //int swarmStrideLevel = 5;

        int phiL2 = getThreadsPerWarpL2();
        int sharedBufferLength = 4;
        int sharedDataSize = sharedBufferLength << phiL2;

        std::stringstream result; if (qG == 1) result << "\
            typedef float T;\n\
            struct Komplex { T r, i; };\n\
            typedef struct Komplex K;\n\
            inline K komplex(T iR, T iI);       inline K komplex(T iR, T iI) { K k; k.r = iR; k.i = iI; return k; };\n\
            inline K unit(int n, int d);\n\
            inline K mul(const K a, const K b); inline K mul(const K a, const K b) { return komplex(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r); };\n\
            inline K add(const K a, const K b); inline K add(const K a, const K b) { return komplex(a.r + b.r, a.i + b.i); }\n\
            inline K unit(int n, int d) { const float frac_PI2_d = " << (2*M_PI) << "f/d; return komplex(native_cos(frac_PI2_d*n), native_sin(frac_PI2_d*n)); };\n;";

        result << "__kernel void contiguousFFT_step" << qG << "(__global float *in, __global float *out) {\n";
        //result << "__local float sharedData[" << sharedDataSize << "];";
        //result << "int globid = get_global_id(0); // = phi*w + v \n";
        result << "int jG = get_global_id(0);\n"; // = phi*w + v \n
          /*  int swarmIxOffset = globid >> " << swarmStrideLevel << ";\n\
            int subSwarmIx = globid & " << ((1 << swarmStrideLevel) - 1) << ";\n\
            for (int swarmIx = 0; swarmIx < " << swarmSize << "; ++swarmIx) {\n\
            int j = subSwarmIx + ((swarmIxOffset*" << swarmSize << " + swarmIx) << " << swarmStrideLevel << ");\n";*/
        result << "if (jG < " << (LG/BGs[q_G])*getWarpsPerKernel(qG) << ") {\n";
//        result << "out[jG] = jG; ";
        printf("(wpk = %i)", getWarpsPerKernel(qG));
        result << "int jM = (jG >> " << phiL2 << ") % " << getWarpsPerKernel(qG) << ";\n";
//        result << "out[jM] = -jM;";
        result << "jG = (jG / " << getWarpsPerKernel(qG) << ") /*+ (jM * " << ((LG/BGs[q_G]) << phiL2) << ")*/;\n";

        result << "\
        int gG = jG/" << LG/NGs[q_G] << ";\n\
        int zG = jG - " << LG/NGs[q_G] << "*gG;\n";

        // M-Level FFT
        for (int qM = 1; qM <= BM.size(); ++qM) { int q_M = qM - 1; result << "{";
            result << "\
            int gM = jM/" << BGs[q_G]/NM[q_M] << ";\
            int zM = jM - " << BGs[q_G]/NM[q_M] << "*gM;";
            std::vector<int> BL = std::vector<int>(ceilog(BM[q_M], 2), 2);
            int kL = BL.size();
            printf("M-level %i: (", qM); for (int b = 0; b < BL.size(); ++b) printf("[%i] ", BL[b]); printf(")");
            std::vector<int> AL, NL;
            computeParameters(AL, BL, NL);

            Array buff0 = Array("K", BM[q_M], true, "buff0_");
            Array buff1 = Array("K", BM[q_M], true, "buff1_");
            Array* buffs[2] = { &buff0, &buff1 };
            result << buff0.getDeclaration() << buff1.getDeclaration();

            // M-Level Read (and pretwiddle)
            if (qM == 1) { // M-level: From global memory
                result << "int readStartOffsetG = zG + " << ((LG/NGs[q_G]) * BGs[q_G]) << "*gG;\n";
                result << "int readStartOffsetM = zM + " << ((BGs[q_G]/NM[q_M]) * BM[q_M]) << "*gM;\n";
                for (int sM = 0; sM < BM[q_M]; ++sM) { result << "{"
                    << "int index = readStartOffsetG + (readStartOffsetM + " << sM*(BGs[q_G]/NM[q_M]) << ")*" << (LG/NGs[q_G]) << ";"
                    << "int ix = ((index >> " << GlobalPlannarLevel << ") << " << (GlobalPlannarLevel + 1) << ") | (index & " << plannarMask << ");"
                    << buff0.getItem(sM)() << ".r = in[ix];"
                    << buff0.getItem(sM)() << ".i = in[ix + " << (1 << GlobalPlannarLevel) << "];"
                    << "}\n";
                }
                if (qG > 1) for (int sM = 0; sM < BM[q_M]; ++sM) {
                    result << buff0.getItem(sM)()
                    << " = mul(" << buff0.getItem(sM)() << ", unit(-(readStartOffsetM + " << sM*(BGs[q_G]/NM[q_M]) << ")*gG, "<< NGs[q_G] << ")" << ");\n";
                }
            }

/*            if ((qM > 1) && (qM < BM.size())) { int q_MPrev = q_M - 1; result << "{"; // Local exchange
                result << "int threadIx = (jG & " << ((1 << phiL2) - 1) << ");";
                result << "int threadSubSetIx = jG ;";
                result << "\
                int gR = gM;\
                int zR = zM;\
                int gW = jM/" << BGs[q_G]/NM[q_M - 1] << ";\
                int zW = jM - " << BGs[q_G]/NM[q_M - 1] << "*gM;";
                for (int y = 0; y < synchronisationCount; ++y) { result << "{";
                    result << "int writeStartOffsetM = " << BGs[q_G]/NM[q_MPrev] << "*gM + zM;\n";
                    for (int hM = 0; hM < BM[q_MPrev]; ++hM) { result
                        << "int index = writeStartOffsetM + " << hM*AM[q_MPrev]*(BGs[q_G]/NM[q_MPrev]) << ";";
                    }
                    result << "}";

                    result << "int readStartOffsetM = zM + " << ((BGs[q_G]/NM[q_M]) * BM[q_M]) << "*gM;\n";
                    for (int sM = 0; sM < BM[q_M]; ++sM) { result
                        << "int index = readStartOffsetM + " << sM*(BGs[q_G]/NM[q_M]) << ";";
                    }
                    for (int sM = 0; sM < BM[q_M]; ++sM) {
                        result << buff0.getItem(sM)()
                        << " = mul(" << buff0.getItem(sM)() << ", unit(" << -sM << "*gM, "<< NMs[q_M] << ")" << ");\n";
                    }
                    result << "}";
                }
                result << "}";
            }*/

            // L-level FFT
            for (int qL = 1; qL <= kL; ++qL) { int q_L = qL - 1;
                Array& source = *buffs[(qL ^ 1) & 1];
                Array& target = *buffs[qL & 1];

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

            // Global Write
            if (qM == kM) { // M-level: To global memory
                result << "int writeStartOffsetG = zG + " <<       LG/NGs[q_G] << "*gG;\n";
                result << "int writeStartOffsetM = zM + " << BGs[q_G]/NM[q_M] << "*gM;\n";
                for (int hM = 0; hM < BM[q_M]; ++hM) { result << "{"
                    << "int index = writeStartOffsetG + (writeStartOffsetM + " << hM*AM[q_M]*(BGs[q_G]/NM[q_M]) << ")*" << AGs[q_G]*(LG/NGs[q_G]) << ";"
                    << "int ix = ((index >> " << GlobalPlannarLevel << ") << " << (GlobalPlannarLevel + 1) << ") | (index & " << plannarMask << ");"
                    << "out[ix] = " << buffs[kL & 1]->getItem(hM)() << ".r;"
                    << "out[ix + " << (1 << GlobalPlannarLevel) << "] = " << buffs[kL & 1]->getItem(hM)() << ".i;"
                    << "}\n";
                }
//                result << "out[2*jM + jG] = writeStartOffsetG;";
  //              result << "out[2*(2 + jM)] = jG;";
             //   result << "out[4 + jM] = jM;";
               // result << "out[6 + jM] = jG;";
            }
            result << "}";
        }

        result << "}}";
//        printf("#####################\n%s\n", result.str().c_str());
        return result.str();
    }
};

#endif // FFT_OPENCL_LAC_H
