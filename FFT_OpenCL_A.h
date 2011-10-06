/*
 * The information in this file is
 * Copyright (C) 2010-2011, Sven De Smet <sven@cubiccarrot.com>
 * and is subject to the terms and conditions of the
 * GNU Lesser General Public License Version 2.1
 * The license text is available from
 * http://www.gnu.org/licenses/lgpl.html
 */

#ifndef FFT_OPENCL_APL_H
#define FFT_OPENCL_APL_H

#define __CL_ENABLE_EXCEPTIONS
#include <OpenCL/opencl.h>
#include "cl.hpp"
#include "FFT.h"
#include "OpenCL_FFT/clFFT.h"

#include "uOpenCL.h"

typedef enum { clFFT_OUT_OF_PLACE, clFFT_IN_PLACE } clFFT_TestType;

typedef struct {
        double real;
        double imag;
} clFFT_ComplexDouble;

typedef struct {
        double *real;
        double *imag;
} clFFT_SplitComplexDouble;

template <class D> class FFT_OpenCL_A : public FFT<D> {
private:
    clFFT_Plan plan;
    ComplexArrayCL<D>* data_in;
    ComplexArrayCL<D>* data_out;
    cl::Context* context;
    bool forward;
    cl::CommandQueue* commandQueue;
public:
    FFT_OpenCL_A(int iSize, bool iForward = true, int iBatchCount = 1) : FFT<D>(iSize, iBatchCount), forward(iForward), context(NULL), commandQueue(NULL) {
        clFFT_DataFormat dataFormat = clFFT_SplitComplexFormat;
        bool planar = (dataFormat == clFFT_SplitComplexFormat);
        this->in = new ComplexArray<D>(this->batchCount*this->size, planar);
        this->out = new ComplexArray<D>(this->batchCount*this->size, planar);

        std::vector<cl::Platform> platforms;
        xCLErr(cl::Platform::get(&platforms));

        std::vector<cl::Device> devicesToUse;
        for (int p = 0; p < (int) platforms.size(); ++p) { CLPlatform platform = CLPlatform(platforms[p]);
            /*printf("== Platform %i: %s ==", p, platform.name().c_str());
            printf("Vendor: %s", platform.vendor().c_str());
            printf("Profile: %s", platform.profile().c_str());
            printf("Version: %s", platform.version().c_str());
            printf("Extensions: %s", platform.extensions().c_str());
*/
            std::vector<cl::Device> devices;
            xCLErr(platforms[p].getDevices(CL_DEVICE_TYPE_GPU, &devices));
       //     qDe bug("%i devices", (int) devices.size());
            for (int d = 0; d < (int) devices.size(); ++d) { CLDevice device = CLDevice(devices[d]);
                printf("[Device: %s, ", device.name().c_str());
                printf("Vendor: %s]", device.vendor().c_str());

  /*              printf("-- Device name: %s --", device.name().c_str());
                printf("Vendor: %s", device.vendor().c_str());
                printf("Global Memory Size: %ld", device.globalMemorySize());
                printf("Local Memory Size: %ld", device.localMemorySize());
                printf("Max Compute Units: %i", device.maxComputeUnits());
                printf("Max Clock Frequency: %i", device.maxClockFrequency());
                printf("Max Work Group Size: %i", device.maxWorkGroupSize(0));
                printf("Preferred Vector Width Float: %i", device.preferredVectorWidthFloat());*/
                if (device.available()) {
                   // printf("Available");
                    devicesToUse.push_back(devices[d]);
                } //else { printf("Not available"); }
            }
        }

        cl_int err;
        context = new cl::Context::Context(devicesToUse, NULL, NULL, NULL, &err);
        xCLErr(err);

        commandQueue = new cl::CommandQueue::CommandQueue(*context, devicesToUse[0], 0, &err);
        xCLErr(err);

        clFFT_TestType testType = clFFT_OUT_OF_PLACE;
        cl_ulong memReq = (testType == clFFT_OUT_OF_PLACE) ? 3 : 2;
        memReq *= this->size*sizeof(clFFT_Complex)*this->batchCount;
        memReq = memReq;
        if(memReq >= CLDevice(devicesToUse[0]).globalMemorySize()) { printf("Insufficient global memory"); } // throw exception

        clFFT_Dim3 dims = { this->size, 1, 1 };

        plan = clFFT_CreatePlan((*context)(), dims, clFFT_1D, dataFormat, &err );
        xCLErr(err);

        //printf("%s", ((cl_fft_plan *) plan)->kernel_string->c_str());
/*        FILE* f = fopen("test.txt", "wb+");
        clFFT_DumpPlan(plan, f);
        fclose(f);
 */       //gMemSize /= (1024*1024);
        //(checkMemRequirements(n, batchSize, testType, gMemSize))
        //err = runTest(n, batchSize, dir, dim, dataFormat, numIter, testType);
        //data = new ComplexArray<float>(length);

        data_in = new ComplexArrayCL<float>(*context, this->batchCount*this->size, planar);
        data_out = (testType == clFFT_OUT_OF_PLACE) ? new ComplexArrayCL<float>(*context, this->batchCount*this->size, planar) : data_in;
    }

    virtual void execute() {
        clFFT_Direction dir = forward ? clFFT_Forward : clFFT_Inverse;

        if (this->size == 1) {
            for (int q = 0; q < this->size; ++q) this->out->setElement(q, this->in->getElement(q));
        } else {
            data_in->enqueueWriteArray(*commandQueue, *this->in);

            clFFT_DataFormat dataFormat = clFFT_SplitComplexFormat;
            bool planar = (dataFormat == clFFT_SplitComplexFormat);
            if (planar) { xCLErr(clFFT_ExecutePlannar((*commandQueue)(), plan, this->batchCount, dir, data_in->getReals(), data_in->getImaginaries(), data_out->getReals(), data_out->getImaginaries(), 0, NULL, NULL)); }
            else { xCLErr(clFFT_ExecuteInterleaved((*commandQueue)(), plan, this->batchCount, dir, data_in->getData(), data_out->getData(), 0, NULL, NULL)); }

            xCLErr(clFinish((*commandQueue)()));

            data_out->enqueueReadArray(*commandQueue, *this->out);
        }
    }

    ~FFT_OpenCL_A() {
        clFFT_DestroyPlan(plan);

        delete context;
        delete commandQueue;
        delete this->in;
        delete this->out;
    }
};

#endif // FFT_OPENCL_APL_H
