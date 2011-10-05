/*
 * The information in this file is
 * Copyright (C) 2010-2011, Sven De Smet <sven@cubiccarrot.com>
 * and is subject to the terms and conditions of the
 * GNU Lesser General Public License Version 2.1
 * The license text is available from
 * http://www.gnu.org/licenses/lgpl.html
 */

#ifndef FFT_OPENCL_CONTIGUOUS_INNERKERNELTESTER_H
#define FFT_OPENCL_CONTIGUOUS_INNERKERNELTESTER_H


template <class D> class FFT_OpenCL_Contiguous_InnerKernelTester : public FFT<D> {
private:
    clFFT_Plan plan;
    ComplexArrayCL<D>* data[2];
    cl::Context* context;
    bool forward;
    cl::CommandQueue* commandQueue;
    cl::Kernel* kernel;
    cl::Program* program;
    cl::Program::Sources* source;
    std::vector<cl::Device> devicesToUse;
    std::string src;
public:
    FFT_OpenCL_Contiguous_InnerKernelTester(int iSize, bool iForward = true, int iBatchCount = 1) : FFT<D>(iSize, iBatchCount), forward(iForward), context(NULL), commandQueue(NULL), kernel(NULL) {
        this->in = new ComplexArray<D>(this->batchCount*this->size, false);
        this->out = new ComplexArray<D>(this->batchCount*this->size, false);

        std::vector<cl::Platform> platforms;
        xCLErr(cl::Platform::get(&platforms));

        for (int p = 0; p < (int) platforms.size(); ++p) { CLPlatform platform = CLPlatform(platforms[p]);
            std::vector<cl::Device> devices;
            xCLErr(platforms[p].getDevices(CL_DEVICE_TYPE_CPU, &devices));
            for (int d = 0; d < (int) devices.size(); ++d) { CLDevice device = CLDevice(devices[d]);
                if (device.available()) devicesToUse.push_back(devices[d]);
            }
        }

        cl_int err;
        context = new cl::Context::Context(devicesToUse, NULL, NULL, NULL, &err);
        xCLErr(err);

        commandQueue = new cl::CommandQueue::CommandQueue(*context, devicesToUse[0], 0, &err);
        xCLErr(err);

        int memReq = this->size * sizeof(clFFT_Complex) * this->batchCount;
        if(memReq >= CLDevice(devicesToUse[0]).globalMemorySize()) { printf("Insufficient global memory"); } // throw exception

        data[0] = new ComplexArrayCL<float>(*context, this->batchCount*this->size, false);
        data[1] = new ComplexArrayCL<float>(*context, this->batchCount*this->size, false);

        int log2Size = (int) (log(this->size)/log(2) + 0.5);
        std::vector<int> BL;
        for (int qL = 1; qL <= log2Size; ++qL) BL.push_back(2);
        src = KernelGenerator::generateKernel(0, 1, BL, this->size, this->size, true);
        //printf("%s", src.c_str());

        source = new cl::Program::Sources(1, std::make_pair(src.c_str(), src.length()));
        program = new cl::Program(*context, *source);
        try {
            program->build(devicesToUse);

            std::stringstream kernelName; kernelName << "contiguousFFT_step0";
            kernel = new cl::Kernel(*program, kernelName.str().c_str(), &err);
            xCLErr(err);
        } catch (cl::Error cle) {
            printf("Error: %s", cle.what());

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
                    printf("Kernel build error:\n%s", build_log);
                    delete[] build_log;
                } catch (cl::Error err) {
                    printf("Kernel build error: unkown (Failed to retrieve build log).");
                    throw err;
                } } else printf("Kernel build successful.");
            }
        }
    }

    virtual void execute() { if (kernel) {
        if (this->size == 1) {
            for (int q = 0; q < this->size; ++q) this->out->setElement(q, this->in->getElement(q));
        } else {
            data[0]->enqueueWriteArray(*commandQueue, *this->in);

            cl::KernelFunctor func = kernel->bind(*commandQueue, cl::NDRange(1), cl::NDRange(1));
            kernel->setArg<cl_mem>(0, data[0]->getData());
            kernel->setArg<cl_mem>(1, data[1]->getData());

            func().wait();

            data[1]->enqueueReadArray(*commandQueue, *this->out);
        }
    } }

    ~FFT_OpenCL_Contiguous_InnerKernelTester() {
        delete context;
        delete commandQueue;
        delete this->in;
        delete this->out;
    }
};

#endif // FFT_OPENCL_CONTIGUOUS_INNERKERNELTESTER_H
