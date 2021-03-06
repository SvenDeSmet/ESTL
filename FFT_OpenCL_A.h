/* The information in this file is
 * Copyright (C) 2011, Sven De Smet <sven@cubiccarrot.com>
 * and is subject to the terms and conditions of the
 * GNU Lesser General Public License Version 2.1
 * The license text is available from
 * http://www.gnu.org/licenses/lgpl.html
 */

#ifndef FFT_OPENCL_APL_H
#define FFT_OPENCL_APL_H

#include "FFT.h"
#include "OpenCL_FFT/clFFT.h"
#include "uOpenCL.h"

typedef enum { clFFT_OUT_OF_PLACE, clFFT_IN_PLACE } clFFT_TestType;
typedef struct { double real; double imag; } clFFT_ComplexDouble;
typedef struct { double *real; double *imag; } clFFT_SplitComplexDouble;

template <class D, clFFT_DataFormat DataFormat, clFFT_TestType TestType> class FFT_OpenCL_A : public FFT<D>, public OpenCLAlgorithm {
private:
    clFFT_Plan plan;
    ComplexArrayCL<D> *data_in, *data_out;
    bool forward;
    SplitInterleavedDataInterface<D>* splitInterleavedDataInterface;
public:
    static streng getName() {
        streng interleaved = (DataFormat == clFFT_SplitComplexFormat) ? "Non-interleaved" : "Interleaved";
        streng place = (TestType == clFFT_OUT_OF_PLACE) ? "Out-of-place" : "In-place";
        return streng("OpenCL FFT A (") + interleaved + streng(", ") + place + streng(")");
    }

    static int getRequiredMemory(int n) { return ((TestType == clFFT_OUT_OF_PLACE) ? 3 : 2)*n*sizeof(clFFT_Complex); }
    static int getMaxBatchSize(int n) { return getGlobalMemory()/getRequiredMemory(n); }

    FFT_OpenCL_A(int iSize, bool iForward = true, int iBatchCount = 1) : FFT<D>(iSize, iBatchCount), OpenCLAlgorithm(), forward(iForward) {
        bool planar = (DataFormat == clFFT_SplitComplexFormat);
        this->dataInterface = splitInterleavedDataInterface = new SplitInterleavedDataInterface<D>(this->batchCount*this->size, planar);

        if (getMaxBatchSize(this->size) < this->batchCount) { printf("Insufficient global memory"); } // throw exception

        cl_int err;
        clFFT_Dim3 dims = { this->size, 1, 1 };
        plan = clFFT_CreatePlan((*context)(), dims, clFFT_1D, DataFormat, &err); xCLErr(err);

        data_in = new ComplexArrayCL<float>(*context, splitInterleavedDataInterface->in);
        data_out = (TestType == clFFT_OUT_OF_PLACE) ? new ComplexArrayCL<float>(*context, splitInterleavedDataInterface->out) : data_in;
    }

    virtual void execute() {
        clFFT_Direction dir = forward ? clFFT_Forward : clFFT_Inverse;

        if (this->size == 1) { splitInterleavedDataInterface->out->setElement(0, splitInterleavedDataInterface->in->getElement(0)); }
        else {
            data_in->enqueueWriteArray(*commandQueue, *splitInterleavedDataInterface->in, true);

            if (DataFormat == clFFT_SplitComplexFormat) {
                xCLErr(clFFT_ExecutePlannar((*commandQueue)(), plan, this->batchCount, dir, data_in->getReals(), data_in->getImaginaries(), data_out->getReals(), data_out->getImaginaries(), 0, NULL, NULL));
            } else {
                xCLErr(clFFT_ExecuteInterleaved((*commandQueue)(), plan, this->batchCount, dir, data_in->getData(), data_out->getData(), 0, NULL, NULL));
            }

            data_out->enqueueReadArray(*commandQueue, *splitInterleavedDataInterface->out, true);
        }
    }

    void printGPUDebugData() {
        printf("\nout: ");
        data_out->enqueueReadArray(*commandQueue, *splitInterleavedDataInterface->out);
        for (int q = 0; q < this->size; ++q) splitInterleavedDataInterface->out->getElement(q).print();
        printf("\nin: ");
        data_in->enqueueReadArray(*commandQueue, *splitInterleavedDataInterface->in);
        for (int q = 0; q < this->size; ++q) splitInterleavedDataInterface->in->getElement(q).print();
    }

    virtual ~FFT_OpenCL_A() {
        clFFT_DestroyPlan(plan);
        if (data_in != data_out) delete data_out;
        delete data_in;
    }
};

#endif // FFT_OPENCL_APL_H
