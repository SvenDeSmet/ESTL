/* The information in this file is
 * Copyright (C) 2011, Sven De Smet <sven@cubiccarrot.com>
 * and is subject to the terms and conditions of the
 * GNU Lesser General Public License Version 2.1
 * The license text is available from
 * http://www.gnu.org/licenses/lgpl.html
 */

#ifndef FFT_OPENCL_CONTIGUOUS_H
#define FFT_OPENCL_CONTIGUOUS_H

#include <vector>
#include <sstream>
#include <string>
#include <exception>

#include "uOpenCL.h"
#include "FFT.h"
#include "tests/Timer.h"
#include <sys/types.h>
#include "KernelGenerator.h"

#define DBG(a) a

#include "SimpleMath.h"

template <class D> class FFT_OpenCL_Contiguous : public FFT<D>, public OpenCLAlgorithm {
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
    static streng getName() { return "Contiguous OpenCL FFT"; }

    std::vector<int> AGs, BGs, NGs;

    FFT_OpenCL_Contiguous(int iSize, bool iForward = true, int iBatchCount = 1) : FFT<D>(iSize, iBatchCount), OpenCLAlgorithm(), forward(iForward) {
        this->dataInterface = arrayDataInterface = new DataInterfaceType(this->batchCount*this->size);

        int memReq = this->size*2*sizeof(D)*this->batchCount;
        if(memReq >= CLDevice(devicesToUse[0]).globalMemorySize()) { printf("Insufficient global memory"); } // throw exception

        data[0] = new ComplexArrayCL<float>(*context, arrayDataInterface->in);
        data[1] = new ComplexArrayCL<float>(*context, arrayDataInterface->out);

        if (this->size > 1) {
            int defaultBG = 16;
            int logBGSize = ceilog(this->size, defaultBG);
            int AG = 1;
            int NG = 1;

            source = new cl::Program::Sources();
            for (int qG = 1; qG <= logBGSize; ++qG) { //printf("qG = %i", qG);
//                int BG = (qG == 1) ? this->size/pow(BG, (logBGSize - 1)) : defaultBG;
                int BG = mIn(defaultBG, this->size/NG);
                NG *= BG;
                //printf("{%i}", BG);
                AGs.push_back(AG); BGs.push_back(BG); NGs.push_back(NG);
                std::vector<int> BL = std::vector<int>(ceilog(BG, 2), 2);
                //printf("Generate Kernel qG = %i, AG = %i, fracLG_NG - %i", qG, AG, this->size/NG);
                src.push_back(generateKernel(qG, AG, BL, this->size, NG, (qG == 1), getSwarmSize(qG), true));
               // printf("@@@@\n%s\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", src[src.size() - 1].c_str());
                source->push_back(std::make_pair(src[src.size() - 1].c_str(), src[src.size() - 1].length()));
                AG *= BG;
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
                        for (int s = 0; s < (int) src.size(); ++s) printf("%s", src[s].c_str());
                        printf("Kernel build error:\n%s", build_log);
                        delete [] build_log;
                    } catch (cl::Error err) { printf("Kernel build error: unkown (Failed to retrieve build log)."); throw err; }
                    } else printf("Kernel build successful.");
                }
            } catch (std::exception e) { printf("%s", e.what()); }

            for (int qG = 1; qG <= logBGSize; ++qG) {
                std::stringstream kernelName; kernelName << "contiguousFFT_step" << qG;
                cl_int err;
                kernels.push_back(new cl::Kernel(*program, kernelName.str().c_str(), &err)); xCLErr(err);
            }
        }

        kernelTimers.resize(kernels.size()); // printf("Initialization complete"); fflush(stdout);
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

    virtual double getTotalComputationFlops(int kernel) { return (5.0*this->size)*(log((double) BGs[kernel])/log(2.0)); }
    virtual streng getKernelInfo(int kernel) { return intToStr(ceilog(BGs[kernel], 2)); }

    void printGPUDebugData(int qG) {
        data[0]->enqueueReadArray(*commandQueue, *arrayDataInterface->out);
        printf("0 qG = %i", qG); for (int q = 0; q < this->size; ++q) arrayDataInterface->out->getElement(q).print();
        data[1]->enqueueReadArray(*commandQueue, *arrayDataInterface->in);
        printf("1 qG = %i", qG); for (int q = 0; q < this->size; ++q) arrayDataInterface->in->getElement(q).print();
    }

    virtual void execute() { //printf("Starting execution with %i kernels", (int) kernels.size()); fflush(stdout);
        if (this->size == 1) { arrayDataInterface->out->setElement(0, arrayDataInterface->in->getElement(0)); }
        else { if (kernels.size() > 0) {
//          for (int q = 0; q < this->size; ++q) this->out->setElement(q, 888); data[1]->enqueueWriteArray(*commandQueue, *this->out);
            data[0]->enqueueWriteArray(*commandQueue, *arrayDataInterface->in);

            kernelEvents.clear();
            for (int qG = 1; qG <= (int) kernels.size(); qG++) { cl::Kernel* kernel = kernels[qG - 1]; //printf("kernel q = %i", qG);
                size_t workGroupSize;
                kernel->getWorkGroupInfo<size_t>(devicesToUse[0], CL_KERNEL_WORK_GROUP_SIZE, &workGroupSize); // printf("[workGroupSize = %i]", workGroupSize);
                int localSize = mIn(this->size/BGs[qG - 1], workGroupSize);

                //printf("<%i %i>", workGroupSize, localSize);
                //printf("wgs = %i --- size = %i -- s = %i -- localSize == %i -- butterFlyCount = %i", workGroupSize, this->size, s, localSize, butterflyCount);
                cl::KernelFunctor func = kernel->bind(*commandQueue, cl::NDRange(ceilint(getSwarmCount(qG), localSize)), cl::NDRange(localSize));
                kernel->setArg<cl_mem>(0, data[(qG ^ 1) & 1]->getData());
                kernel->setArg<cl_mem>(1, data[(qG ^ 0) & 1]->getData());

 //printf("Before execution"); printGPUDebugData(qG);
                try { kernelEvents.push_back(func()); } // enqueue kernel
                catch (cl::Error e) { printf("CL Exception"); CLException cle = CLException(e); cle.handle(); fflush(stdout); }
                catch (...) { printf("Unknown exception"); fflush(stdout); }
//                kernelEvents[kernelEvents.size() - 1].wait(); printf("After execution"); printGPUDebugData(qG); fflush(stdout);
            } // printf("Computation ends..."); fflush(stdout);

            data[kernels.size() & 1]->enqueueReadArray(*commandQueue, *arrayDataInterface->out);

            kernelEvents[kernels.size() - 1].wait();
            if (timerComputation) timerComputation->addRun(getTime(0, true, kernels.size() - 1, false));
            for (int qG = 1; qG <= (int) kernels.size(); ++qG) kernelTimers[qG - 1].addRun(getTime(qG - 1, true, qG - 1, false));

//            printf("out"); for (int q = 0; q < this->size; ++q) this->out->getElement(q).print();
//            data[(kernels.size() & 1) ^ 1]->enqueueReadArray(*commandQueue, *this->in);
//            printf("in"); for (int q = 0; q < this->size; ++q) this->in->getElement(q).print();
        } }
    }

    static streng generateKernel(int kernelIx, int Aq, std::vector<int> BL, int LG, int NG, bool includeCommonDefs, int swarmSize = 1, bool preTwiddle = false) {
        int swarmStrideLevel = 5;
        int kL = BL.size();
        int LL = 1;
        for (int qL = 1; qL <= kL; ++qL) LL *= BL[qL - 1];
        std::vector<int> AL, NL;
        int b = 1;
        for (int qL = 1; qL <= kL; ++qL) {
            AL.push_back(b);
            b *= BL[qL - 1];
            NL.push_back(b);
        }
        int Bq = LL;
        std::stringstream komplex; komplex << "\n\
    typedef float T;\n\
    struct Komplex { T r, i; };\n\
    typedef struct Komplex K;\n\
    inline K komplex(T iR, T iI);\n\
    inline K unit(int n, int d);\n\
    inline K mul(const K a, const K b);\n\
    inline K add(const K a, const K b);\n\
    inline K komplex(T iR, T iI) { K k; k.r = iR; k.i = iI; return k; };\n\
    inline K unit(int n, int d) { const float frac_PI2_d = " << (2*M_PI) << "f/d; return komplex(native_cos(frac_PI2_d*n), native_sin(frac_PI2_d*n)); };\n\
    inline K unitC(int n, const int d) { const float dr = 1.f/d; const float frac_PI2_d = " << (2*M_PI) << "f*dr; return komplex(native_cos(frac_PI2_d*n), native_sin(frac_PI2_d*n)); };\n\
    inline K mul(const K a, const K b) { return komplex(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r); };\n\
    inline K add(const K a, const K b) { return komplex(a.r + b.r, a.i + b.i); };\n;";
        std::stringstream kernelhead; kernelhead << "__kernel void contiguousFFT_step" << kernelIx << "(__global float *in, __global float *out) {\n";
        std::stringstream result;
        result << "int globid = get_global_id(0); // = phi*w + v \n\
        //int swarmIxOffset = globid >> " << swarmStrideLevel << ";\n\
        //int subSwarmIx = globid & " << ((1 << swarmStrideLevel) - 1) << ";\n\
        //for (int swarmIx = 0; swarmIx < " << swarmSize << "; ++swarmIx) {\n\
        //int j = subSwarmIx + ((swarmIxOffset*" << swarmSize << " + swarmIx) << " << swarmStrideLevel << ");\n";
        result << "int j = get_global_id(0);"; // = phi*w + v \n

        Array buff0 = Array("K", LL, true, "buff0_");
        Array buff1 = Array("K", LL, true, "buff1_");
        Array* buffs[2] = { &buff0, &buff1 };
        result << buff0.getDeclaration() << "\n" << buff1.getDeclaration() << "\n";

        result << "  if (j < " << LG/LL << ") {\n\
        int gG = " << IntegerDivision("j", LG/NG)() << ";\n\
        int zG = j - " << LG/NG << "*gG;\n"; // zG = j % " << LG/NG << ";\n";
        // Load data
        result << "int readStartOffset = zG + " << (LG/NG) * Bq << "*gG;";
        int plannarMask = (1 << GlobalPlannarLevel) - 1;
        for (int sG = 0; sG < Bq; ++sG) { result << "{"
            << "int index = readStartOffset + " << sG*(LG/NG) << ";"
            << "int ix = ((index >> " << GlobalPlannarLevel << ") << " << (GlobalPlannarLevel + 1) << ") | (index & " << plannarMask << ");"
            << buff0.getItem(sG).getRepresentation() << ".r = in[ix];"
            << buff0.getItem(sG).getRepresentation() << ".i = in[ix + " << (1 << GlobalPlannarLevel) << "];"
            << "}";
        }
        if (preTwiddle) for (int sG = 0; sG < Bq; ++sG) {
            result << buff0.getItem(sG).getRepresentation()
            << " = mul(" << buff0.getItem(sG).getRepresentation() << ", unitC(" << -sG << "*gG, "<< NG << ")" << ");\n";
        }

        // Local FFT
        for (int qL = 1; qL <= kL; ++qL) {
            Array& source = *buffs[(qL ^ 1) & 1];
            Array& target = *buffs[qL & 1];

            std::stringstream subkernel;
            for (int gL = 0; gL < AL[qL - 1]; ++gL) { subkernel << "{";
                for (int zL = 0; zL < (LL/NL[qL - 1]); ++zL) { subkernel << "{";
                    for (int sL = 0; sL < BL[qL - 1]; ++sL) {
                        subkernel << "const K s" << sL << " = "
                        << KomplexConstMultiplication(KomplexUnit(-gL*sL, NL[qL - 1]), source.getItem((LL/NL[qL - 1])*(BL[qL - 1] * gL + sL) + zL)).getRepresentation() << ";";
                    }

                    for (int hL = 0; hL < BL[qL - 1]; ++hL) {
                        subkernel << target.getItem((LL/NL[qL - 1])*(gL + AL[qL - 1]*hL) + zL).getRepresentation() << " = ";
                        for (int sL = 0; sL < BL[qL - 1] - 1; ++sL) subkernel << "add(";
                        for (int sL = 0; sL < BL[qL - 1]; ++sL) {
                            if (sL > 0) subkernel << ", ";
                            subkernel << KomplexConstMultiplication(KomplexUnit(-hL*sL, BL[qL - 1]), streng("s") + intToStr(sL)).getRepresentation();
                            if (sL > 0) subkernel << ")";
                        }
                        subkernel << ";";
                    }
                    subkernel << "}";
                }
                subkernel << "}";
            }
            result << subkernel.str();
        }

        // Write results
        result << "int writeStartOffset = " << LG/NG << "*gG + zG;\n";
        for (int h = 0; h < Bq; ++h) { result << "{"
            << "int index = writeStartOffset + " << h*Aq*(LG/NG) << ";"
            << "int ix = ((index >> " << GlobalPlannarLevel << ") << " << (GlobalPlannarLevel + 1) << ") | (index & " << plannarMask << ");"
            << "out[ix] = " << buffs[kL & 1]->getItem(h).getRepresentation() << ".r;"
            << "out[ix + " << (1 << GlobalPlannarLevel) << "] = " << buffs[kL & 1]->getItem(h).getRepresentation() << ".i;"
            << "}";
        }
        result << "}}";

        std::stringstream src;
        if (includeCommonDefs) src << komplex.str();
        src << kernelhead.str() << result.str();
        return src.str();
    }

    virtual ~FFT_OpenCL_Contiguous() {
        for (int d = 0; d < 2; ++d) delete data[d];
        kernelEvents.clear();
        for (int k = 0; k < (int) kernels.size(); ++k) delete kernels[k];
        delete source;
        delete program;
    }
};

#endif // FFT_OPENCL_CONTIGUOUS_H
