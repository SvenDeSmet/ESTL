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
#include <sys/types.h>

#include "tests/Timer.h"
#include "KernelGenerator.h"
#include "OpenCLFFTAlgorithm.h"

#define DBG(a) a

#include "SimpleMath.h"

template <class D> class FFT_OpenCL_Contiguous : public OpenCLFFTAlgorithm<D> {
public:
    typedef PlannarizedDataInterface<D> DataInterfaceType;
    //typedef SplitInterleavedDataInterface<D> DataInterfaceType;
private:
    ComplexArrayCL<D>* data[2];
    bool forward, debug;
    std::vector<cl::Kernel*> kernels;
    std::vector<cl::Event> kernelEvents;
    CLProgram* program;
    std::vector<streng> src;
    DataInterfaceType* arrayDataInterface;
public:
    static streng getName() { return "Contiguous OpenCL FFT"; }

    std::vector<int> AGs, BGs, NGs;

    static int getRequiredMemory(int n) { return 3*sizeof(Complex<D>)*n; }
    static int getMaxBatchSize(int n) { return OpenCLAlgorithm::getGlobalMemory()/getRequiredMemory(n); }

    FFT_OpenCL_Contiguous(int iSize, bool iForward = true, int iBatchCount = 1) : OpenCLFFTAlgorithm<D>(iSize, iBatchCount), forward(iForward) {
        this->dataInterface = arrayDataInterface = new DataInterfaceType(this->batchCount*this->size);

        if (getMaxBatchSize(this->size) < this->batchCount) { printf("Insufficient global memory"); } // throw exception

        data[0] = new ComplexArrayCL<float>(*this->context, arrayDataInterface->in);
        data[1] = new ComplexArrayCL<float>(*this->context, arrayDataInterface->out);

        if (this->size > 1) {
            int defaultBGl2 = 4;
            int log2Size = ceilog(this->size, 2);
            int stepsG = (log2Size + (defaultBGl2 - 1))/defaultBGl2;

            BGs = std::vector<int>(log2Size/defaultBGl2, 1 << defaultBGl2);
            if ((log2Size % defaultBGl2) > 0) BGs.push_back(1 << (log2Size % defaultBGl2));
            this->computeParameters(AGs, BGs, NGs);
            for (int qG = 1; qG <= stepsG; ++qG) src.push_back(this->generateKernel(qG - 1));

            program = new CLProgram(*this->context, src, this->devicesToUse);
            for (int qG = 1; qG <= stepsG; ++qG) kernels.push_back(program->getKernel(streng("contiguousFFT_step") + intToStr(qG)));
        }

        debug = false;

        this->kernelTimers.resize(kernels.size()); // printf("Initialization complete"); fflush(stdout);
    }

    double getTime(int first, bool firstStart, int last, bool lastStart) {
        cl_ulong start, end;
        clGetEventProfilingInfo(kernelEvents[last](), lastStart ? CL_PROFILING_COMMAND_START : CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        clGetEventProfilingInfo(kernelEvents[first](), firstStart ? CL_PROFILING_COMMAND_START : CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &start, NULL);
        return (end - start) * 1.0e-9f;
    }

    void printGPUDebugData(int qG) {
        data[(qG + 1) % 2]->enqueueReadArray(*this->commandQueue, *arrayDataInterface->out);
        printf("\nqG = %i", qG); for (int q = 0; q < this->size * this->batchCount; ++q) { if ((q % this->size) == 0) printf("\n");
            arrayDataInterface->out->getElement(q).print();
        }
        //data[1]->enqueueReadArray(*commandQueue, *arrayDataInterface->in);
        //printf("1 qG = %i", qG); for (int q = 0; q < this->size; ++q) arrayDataInterface->in->getElement(q).print();
    }

    int getSwarmSize(int qG) { return 1; } //mAx(1, mIn(16, mIn(this->size/NGs[qG - 1], getButterflyCount(qG)))); }
    int getButterflyCount(int qG) { return this->size/BGs[qG - 1]; }
    int getSwarmCount(int qG) { return getButterflyCount(qG)/getSwarmSize(qG); }
    int getWorkItemsPerKernel(int qG) { return getSwarmCount(qG) * this->batchCount; }

    virtual double getTotalComputationFlops(int kernel) { return (5.0*this->size)*(log((double) BGs[kernel])/log(2.0)) * this->batchCount; }
    virtual streng getKernelInfo(int kernel) { return intToStr(ceilog(BGs[kernel], 2)); }

    virtual void execute() { //printf("Starting execution with %i kernels", (int) kernels.size()); fflush(stdout);
        if (this->size == 1) { arrayDataInterface->out->setElement(0, arrayDataInterface->in->getElement(0)); }
        else { if (kernels.size() > 0) {
            if (!forward) {
                for (int d = 0; d < this->size; ++d) arrayDataInterface->in->setElement(d, arrayDataInterface->in->getElement(d).getConjugate());
            }
            if (debug) for (int q = 0; q < this->size * this->batchCount; ++q) arrayDataInterface->out->setElement(q, 888); data[1]->enqueueWriteArray(*this->commandQueue, *arrayDataInterface->out);
            if (this->timerTotal) this->timerTotal->resume();
            data[0]->enqueueWriteArray(*this->commandQueue, *arrayDataInterface->in);

            kernelEvents.clear();
            for (int qG = 1; qG <= (int) kernels.size(); qG++) { cl::Kernel* kernel = kernels[qG - 1]; //printf("kernel q = %i", qG);
                size_t workGroupSize;
                kernel->getWorkGroupInfo<size_t>(this->devicesToUse[0], CL_KERNEL_WORK_GROUP_SIZE, &workGroupSize); // printf("[workGroupSize = %i]", workGroupSize);
                int localSize = mIn(getWorkItemsPerKernel(qG), workGroupSize);

            //    printf("<%i>", getWorkItemsPerKernel(qG));
                //printf("wgs = %i --- size = %i -- s = %i -- localSize == %i -- butterFlyCount = %i", workGroupSize, this->size, s, localSize, butterflyCount);
                cl::KernelFunctor func = kernel->bind(*this->commandQueue, cl::NDRange(ceilint(getWorkItemsPerKernel(qG), localSize)), cl::NDRange(localSize));
                kernel->setArg<cl_mem>(0, data[(qG ^ 1) & 1]->getData());
                kernel->setArg<cl_mem>(1, data[(qG ^ 0) & 1]->getData());

// printGPUDebugData(qG);
                try { kernelEvents.push_back(func()); } // enqueue kernel
                catch (cl::Error e) { printf("CL Exception"); CLException cle = CLException(e); cle.handle(); fflush(stdout); }
                catch (...) { printf("Unknown exception"); fflush(stdout); }
//                kernelEvents[kernelEvents.size() - 1].wait(); printf("After execution"); printGPUDebugData(qG); fflush(stdout);
            } // printf("Computation ends..."); fflush(stdout);
//printGPUDebugData(kernels.size() + 1);

            data[kernels.size() & 1]->enqueueReadArray(*this->commandQueue, *arrayDataInterface->out);

            kernelEvents[kernels.size() - 1].wait();
            if (this->timerComputation) this->timerComputation->addRun(getTime(0, true, kernels.size() - 1, false));
            for (int qG = 1; qG <= (int) kernels.size(); ++qG) this->kernelTimers[qG - 1].addRun(getTime(qG - 1, true, qG - 1, false));
            if (this->timerTotal) this->timerTotal->suspend();

            if (!forward) {
                for (int d = 0; d < this->size; ++d) arrayDataInterface->out->setElement(d, arrayDataInterface->out->getElement(d).getConjugate());
            }

//            printf("out"); for (int q = 0; q < this->size; ++q) this->out->getElement(q).print();
//            data[(kernels.size() & 1) ^ 1]->enqueueReadArray(*commandQueue, *this->in);
//            printf("in"); for (int q = 0; q < this->size; ++q) this->in->getElement(q).print();
        } }
    }

    virtual streng generateKernel(int q_G) { int qG = q_G + 1;
        //int swarmStrideLevel = 5;
        int LG = this->size;
        std::vector<int> AL, NL, BL = std::vector<int>(ceilog(BGs[q_G], 2), 2);
        this->computeParameters(AL, BL, NL);
        int kL = BL.size();

        std::stringstream result;
        if (qG == 1) result << KomplexMath::getDeclarations();
        result << "__kernel void contiguousFFT_step" << qG << "(__global float *in, __global float *out) {\n";
        //int swarmIxOffset = globid >> " << swarmStrideLevel << ";\n
        //int subSwarmIx = globid & " << ((1 << swarmStrideLevel) - 1) << ";\n
        //for (int swarmIx = 0; swarmIx < " << swarmSize << "; ++swarmIx) {\n
        //int j = subSwarmIx + ((swarmIxOffset*" << swarmSize << " + swarmIx) << " << swarmStrideLevel << ");\n";
        result << "int globid = get_global_id(0);"; // = phi*w + v \n

        Array buff0 = Array("K", BGs[q_G], true, "buff0_");
        Array buff1 = Array("K", BGs[q_G], true, "buff1_");
        PlannarizedComplexCLArray in = PlannarizedComplexCLArray(GlobalPlannarLevel, "in", LG);
        PlannarizedComplexCLArray out = PlannarizedComplexCLArray(GlobalPlannarLevel, "out", LG);
        Array* buffs[2] = { &buff0, &buff1 };
        result << buff0.getDeclaration() << "\n" << buff1.getDeclaration() << "\n";

        result << "  if (globid < " << getWorkItemsPerKernel(qG) << ") {\n\
        int batchIx = (globid/" << getSwarmCount(qG) << ");\
        int j = globid - batchIx*" << getSwarmCount(qG) << ";\
        int gG = " << IntegerDivision("j", LG/NGs[q_G])() << ";\n\
        int zG = j - " << LG/NGs[q_G] << "*gG + batchIx*" << LG << ";\n"; // zG = j % " << LG/NGs[q_G] << ";\n";

        result << "int readStartOffset = zG + " << (LG/NGs[q_G]) * BGs[q_G] << "*gG;";
        for (int sG = 0; sG < BGs[q_G]; ++sG) result << in.assignToItem("readStartOffset + " + intToStr(sG*(LG/NGs[q_G])), Array(buff0[sG]()))(); // Load data
        if (qG > 1) for (int sG = 0; sG < BGs[q_G]; ++sG) result << buff0[sG]() << " = mul(" << buff0[sG]() << ", unit(" << -sG << "*gG, "<< NGs[q_G] << ")" << ");\n";

        result << this->generateLocalFFTKernel(buffs, AL, BL, NL, BGs[q_G]); // Local FFT

        result << "int writeStartOffset = zG + " << LG/NGs[q_G] << "*gG;";
        for (int h = 0; h < BGs[q_G]; ++h) result << out.assignFromItem("writeStartOffset + " + intToStr(h*AGs[q_G]*(LG/NGs[q_G])), Array((*buffs[kL & 1])[h]()))(); // Write results
        result << "}}";

        return result.str();
    }

    virtual ~FFT_OpenCL_Contiguous() {
        for (int d = 0; d < 2; ++d) delete data[d];
        kernelEvents.clear();
        for (int k = 0; k < (int) kernels.size(); ++k) delete kernels[k];
        delete program;
    }
};

#endif // FFT_OPENCL_CONTIGUOUS_H
