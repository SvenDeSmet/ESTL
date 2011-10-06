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


#ifndef FFT_OPENCL_CONTIGUOUS_H
#define FFT_OPENCL_CONTIGUOUS_H

#include <vector.h>
#include <sstream>
#include <string>
#include <exception>

#include "uOpenCL.h"
#include "FFT.h"
#include "ContiguousKernelGenerator.h"
#include <mach/mach_time.h>
#include "tests/Timer.h"
#include <sys/types.h>

#define DBG(a) a

template <class D> class FFT_OpenCL_Contiguous : public FFT<D> {
private:
    ComplexArrayCL<D>* data[2];
    cl::Context* context;
    bool forward;
    cl::CommandQueue* commandQueue;
    std::vector<cl::Kernel*> kernels;
    cl::Program* program;
    cl::Program::Sources* source;
    std::vector<cl::Device> devicesToUse;
    std::vector<streng> src;
    std::vector<int> AGs, BGs, NGs;
    PlannarizedDataInterface<D>* plannarizedDataInterface;
public:
    Timer* timerComputation;

    FFT_OpenCL_Contiguous(int iSize, bool iForward = true, int iBatchCount = 1) : FFT<D>(iSize, iBatchCount), forward(iForward), context(NULL), commandQueue(NULL), timerComputation(NULL) {
        this->dataInterface = plannarizedDataInterface = new PlannarizedDataInterface<D>(this->batchCount*this->size);

//        printf("Retrieving platforms"); fflush(stdout);

        std::vector<cl::Platform> platforms;
        xCLErr(cl::Platform::get(&platforms));

//        devicesToUse = new std::vector<cl::Device>();
        for (int p = 0; p < (int) platforms.size(); ++p) { CLPlatform platform = CLPlatform(platforms[p]);
            /*printf("== Platform %i: %s ==", p, platform.name().c_str());
            printf("Vendor: %s", platform.vendor().c_str());
            printf("Profile: %s", platform.profile().c_str());
            printf("Version: %s", platform.version().c_str());
            printf("Extensions: %s", platform.extensions().c_str());
*/
            std::vector<cl::Device> devices;
//            xCLErr(platforms[p].getDevices(CL_DEVICE_TYPE_CPU, &devices));
            xCLErr(platforms[p].getDevices(CL_DEVICE_TYPE_GPU, &devices));
       //     qDe bug("%i devices", (int) devices.size());
            for (int d = 0; d < (int) devices.size(); ++d) { CLDevice device = CLDevice(devices[d]);
                printf("[Device: %s, ", device.name().c_str());
                printf("Vendor: %s]", device.vendor().c_str());
           /*   printf("Global Memory Size: %ld", device.globalMemorySize());
                printf("Local Memory Size: %ld", device.localMemorySize());
                printf("Max Compute Units: %i", device.maxComputeUnits());
                printf("Max Clock Frequency: %i", device.maxClockFrequency());
                printf("Max Work Group Size: %i", device.maxWorkGroupSize(0));*/
/*                printf("Preferred Vector Width Float: %i", device.preferredVectorWidthFloat());
              */  if (device.available()) {
                   // printf("Available");
                    devicesToUse.push_back(devices[d]);
                } //else { printf("Not available"); }
            }
        }

  //      printf("Creating context..."); fflush(stdout);

        cl_int err;
        context = new cl::Context::Context(devicesToUse, NULL, NULL, NULL, &err);
        xCLErr(err);

//printf("Creating command queue..."); fflush(stdout);
        commandQueue = new cl::CommandQueue::CommandQueue(*context, devicesToUse[0], CL_QUEUE_PROFILING_ENABLE, &err);
        xCLErr(err);

        int memReq = this->size*sizeof(clFFT_Complex)*this->batchCount;
        if(memReq >= CLDevice(devicesToUse[0]).globalMemorySize()) { printf("Insufficient global memory"); } // throw exception

        data[0] = new ComplexArrayCL<float>(*context, plannarizedDataInterface->in);
        data[1] = new ComplexArrayCL<float>(*context, plannarizedDataInterface->out);

        if (this->size > 1) {
            int BG = 16;
            int logBGSize = 0;
            for (int a = 1; a < this->size; a *= BG) logBGSize++;
            int AG = 1;
            int NG = 1;

            source = new cl::Program::Sources();
            for (int qG = 1; qG <= logBGSize; ++qG) { //printf("qG = %i", qG);
                int BGq = BG;
                if ((this->size/NG) < BGq) BGq = (this->size/NG);
                NG *= BGq;
                AGs.push_back(AG);
                BGs.push_back(BGq);
                NGs.push_back(NG);
                std::vector<int> BL;
                int log2Size = 0;
                for (int a = 1; a < BGq; a *= 2) log2Size++;
                for (int qL = 1; qL <= log2Size; ++qL) BL.push_back(2);
                //printf("Generate Kernel qG = %i, AG = %i, fracLG_NG - %i", qG, AG, this->size/NG);
                src.push_back(KernelGenerator::generateKernel(qG, AG, BL, this->size, NG, (qG == 1), true));
                //printf("%s\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", src[src.size() - 1].c_str());
                source->push_back(std::make_pair(src[src.size() - 1].c_str(), src[src.size() - 1].length()));
                AG *= BGq;
            }

            program = new cl::Program(*context, *source);
            try { program->build(devicesToUse, "-cl-mad-enable"); }
            catch (cl::Error cle) {
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
                        for (int s = 0; s < src.size(); ++s) {
                            printf("%s", src[s].c_str());
                        }
                        printf("Kernel build error:\n%s", build_log);
                        delete[] build_log;
                    } catch (cl::Error err) {
                        printf("Kernel build error: unkown (Failed to retrieve build log).");
                        throw err;
                    } } else printf("Kernel build successful.");
                }
            } catch (std::exception e) { printf("%s", e.what()); }

            for (int qG = 1; qG <= logBGSize; ++qG) {
                std::stringstream kernelName; kernelName << "contiguousFFT_step" << qG;
                //printf("kernel %i", qG); fflush(stdout);
                kernels.push_back(new cl::Kernel(*program, kernelName.str().c_str(), &err));
                xCLErr(err);
            }
        }

//        printf("Initialization complete"); fflush(stdout);
    }

    virtual void execute() { //printf("Starting execution with %i kernels", (int) kernels.size()); fflush(stdout);
        if (this->size == 1) { plannarizedDataInterface->out->setElement(0, plannarizedDataInterface->in->getElement(0)); }
        else { if (kernels.size() > 0) {
            //printf("Kernels available..."); fflush(stdout);
//            for (int q = 0; q < this->size; ++q) this->out->setElement(q, 888);
//            data[1]->enqueueWriteArray(*commandQueue, *this->out);
            data[0]->enqueueWriteArray(*commandQueue, *plannarizedDataInterface->in);

            std::vector<cl::Event> kernelEvents;
            for (int qG = 1; qG <= kernels.size(); ++qG) { cl::Kernel* kernel = kernels[qG - 1]; //printf("kernel q = %i", qG);
                size_t workGroupSize;
                kernel->getWorkGroupInfo<size_t>(devicesToUse[0], CL_KERNEL_WORK_GROUP_SIZE, &workGroupSize);
      //  printf("[workGroupSize = %i]", workGroupSize);

                int butterflyCount = this->size/BGs[qG - 1];
                int s = this->size/BGs[qG - 1];
                int localSize = (s > workGroupSize) ? workGroupSize : s;
                butterflyCount = localSize*((butterflyCount + (localSize - 1))/localSize);

                //printf("wgs = %i --- size = %i -- s = %i -- localSize == %i -- butterFlyCount = %i", workGroupSize, this->size, s, localSize, butterflyCount);
                cl::KernelFunctor func = kernel->bind(*commandQueue, cl::NDRange(butterflyCount), cl::NDRange(localSize));
                kernel->setArg<cl_mem>(0, data[(qG ^ 1) & 1]->getData());
                kernel->setArg<cl_mem>(1, data[(qG ^ 0) & 1]->getData());

   /*         printf("Before execution");
            data[0]->enqueueReadArray(*commandQueue, *this->out);
            printf("0 qG = %i", qG);
           for (int q = 0; q < this->size; ++q) this->out->getElement(q).print();
            data[1]->enqueueReadArray(*commandQueue, *this->in);
            printf("1 qG = %i", qG);
            for (int q = 0; q < this->size; ++q) this->in->getElement(q).print();
*/
                try { kernelEvents.push_back(func()); } // enqueue kernel
                catch (cl::Error e) { printf("CL Exception"); CLException cle = CLException(e); cle.handle(); fflush(stdout); }
                catch (...) { printf("Unknown exception"); fflush(stdout); }

//    xCLErr(clFinish((*commandQueue)()));
  /*          printf("After execution");
            data[0]->enqueueReadArray(*commandQueue, *this->out);
            printf("0 qG = %i", qG);
            for (int q = 0; q < this->size; ++q) this->out->getElement(q).print();
            data[1]->enqueueReadArray(*commandQueue, *this->in);
            printf("1 qG = %i", qG);
            for (int q = 0; q < this->size; ++q) this->in->getElement(q).print();
*/
  //              kernel->getWorkGroupInfo<size_t>(devicesToUse[0], CL_KERNEL_WORK_GROUP_SIZE, &workGroupSize);
//                printf("workGroupSize = %i", workGroupSize);
              //  if (qG == kernels.size()) e.wait();
            }


//            printf("Computation ends..."); fflush(stdout);
            data[kernels.size() & 1]->enqueueReadArray(*commandQueue, *plannarizedDataInterface->out);

            cl_ulong start, end;
            clGetEventProfilingInfo(kernelEvents[kernels.size() - 1](), CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            clGetEventProfilingInfo(kernelEvents[0](), CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (timerComputation) timerComputation->addRun((end - start) * 1.0e-9f);
//            printf("out");
//            for (int q = 0; q < this->size; ++q) this->out->getElement(q).print();
//            data[(kernels.size() & 1) ^ 1]->enqueueReadArray(*commandQueue, *this->in);
//            printf("in");
//            for (int q = 0; q < this->size; ++q) this->in->getElement(q).print();
//            printf("f");
        } }
    }

    virtual ~FFT_OpenCL_Contiguous() {
        for (int d = 0; d < 2; ++d) delete data[d];
        for (int k = 0; k < (int) kernels.size(); ++k) delete kernels[k];
        delete source;
        delete program;
        delete context;
        delete commandQueue;
    }
};

#endif // FFT_OPENCL_CONTIGUOUS_H
