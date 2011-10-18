/* The information in this file is
 * Copyright (C) 2011, Sven De Smet <sven@cubiccarrot.com>
 * and is subject to the terms and conditions of the
 * GNU Lesser General Public License Version 2.1
 * The license text is available from
 * http://www.gnu.org/licenses/lgpl.html
 */

#ifndef UOPENCL_H
#define UOPENCL_H

#include <string>
#define __CL_ENABLE_EXCEPTIONS
#define CL_LOG_ERRORS stdout

/*
#ifdef MAC
#include <OpenCL/opencl.h>
#endif
#ifdef LINUX
#include </opt/AMDAPP/include/CL/opencl.h>
#endif
*/
#include "cl.hpp"
#include <exception>
#include <string>
#include "tests/Timer.h"
#define xCLErr(result) { if (result != CL_SUCCESS) { printf("Exception"); fflush(stdout); throw CLException(result); } }

class CLException : public std::exception {
private:
    cl::Error e;
public:
    CLException(cl::Error iE) : e(iE) { }

  virtual const char* what() const throw() { return handle().c_str(); }

    std::string handle() const {
        std::string msg = "[";
        switch (e.err()) {
            case CL_INVALID_COMMAND_QUEUE: msg += "CL_INVALID_COMMAND_QUEUE"; break;
            case CL_INVALID_CONTEXT: msg += "CL_INVALID_CONTEXT"; break;
            case CL_INVALID_MEM_OBJECT: msg += "CL_INVALID_MEM_OBJECT"; break;
            case CL_INVALID_VALUE: msg += "CL_INVALID_VALUE"; break;
            case CL_INVALID_PROGRAM_EXECUTABLE: msg += "CL_INVALID_PROGRAM_EXECUTABLE"; break;
            case CL_INVALID_KERNEL: msg += "CL_INVALID_KERNEL"; break;
            case CL_INVALID_KERNEL_ARGS: msg += "CL_INVALID_KERNEL_ARGS"; break;
            case CL_INVALID_WORK_DIMENSION: msg += "CL_INVALID_WORK_DIMENSION"; break;
            case CL_INVALID_WORK_GROUP_SIZE: msg += "CL_INVALID_WORK_GROUP_SIZE"; break;
            case CL_INVALID_WORK_ITEM_SIZE: msg += "CL_INVALID_WORK_ITEM_SIZE"; break;
            case CL_INVALID_GLOBAL_OFFSET: msg += "CL_INVALID_GLOBAL_OFFSET"; break;
            case CL_OUT_OF_RESOURCES: msg += "CL_OUT_OF_RESOURCES"; break;
            case CL_MEM_OBJECT_ALLOCATION_FAILURE: msg += "CL_MEM_OBJECT_ALLOCATION_FAILURE"; break;
            case CL_INVALID_EVENT_WAIT_LIST: msg += "CL_INVALID_EVENT_WAIT_LIST"; break;
            case CL_OUT_OF_HOST_MEMORY: msg += "CL_OUT_OF_HOST_MEMORY"; break;
        }
        msg += " (CL Exception)]";
        printf("%s", msg.c_str());
        fflush(stdout);
        return msg;
    }
};


class CLPlatform {
private:
    cl::Platform platform;
public:
    CLPlatform(cl::Platform iPlatform) : platform(iPlatform) { }

    std::string name() {
        std::string result;
        xCLErr(platform.getInfo(CL_PLATFORM_NAME, &result));
        return result;
    }

    std::string vendor() {
        std::string result;
        xCLErr(platform.getInfo(CL_PLATFORM_VENDOR, &result));
        return result;
    }

    std::string profile() {
        std::string result;
        xCLErr(platform.getInfo(CL_PLATFORM_PROFILE, &result));
        return result;
    }

    std::string version() {
        std::string result;
        xCLErr(platform.getInfo(CL_PLATFORM_VERSION, &result));
        return result;
    }

    std::string extensions() {
        std::string result;
        xCLErr(platform.getInfo(CL_PLATFORM_EXTENSIONS, &result));
        return result;
    }
};

class CLDevice {
private:
    cl::Device device;
public:
    CLDevice(cl::Device iDevice) : device(iDevice) { }

    std::string name() {
        std::string result;
        xCLErr(device.getInfo<std::string>(CL_DEVICE_NAME, &result));
        return result;
    }

    std::string vendor() {
        std::string result;
        xCLErr(device.getInfo<std::string>(CL_DEVICE_VENDOR, &result));
        return result;
    }

    bool available() {
        cl_bool result;
        xCLErr(device.getInfo<cl_bool>(CL_DEVICE_AVAILABLE, &result));
        return result == CL_TRUE;
    }

    cl_ulong globalMemorySize() {
        cl_ulong result;
        xCLErr(device.getInfo<cl_ulong>(CL_DEVICE_GLOBAL_MEM_SIZE, &result));
        return result;
    }

    cl_ulong localMemorySize() {
        cl_ulong result;
        xCLErr(device.getInfo<cl_ulong>(CL_DEVICE_LOCAL_MEM_SIZE, &result));
        return result;
    }

    cl_uint maxComputeUnits() {
        cl_uint result;
        xCLErr(device.getInfo<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS, &result));
        return result;
    }

    size_t maxWorkGroupSize(int dim) {
        size_t result[3];
        xCLErr(device.getInfo<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE, &result[0]));
        return result[dim];
    }

    cl_uint maxClockFrequency() {
        cl_uint result;
        xCLErr(device.getInfo<cl_uint>(CL_DEVICE_MAX_CLOCK_FREQUENCY, &result));
        return result;
    }

    cl_uint preferredVectorWidthFloat() {
        cl_uint result;
        xCLErr(device.getInfo<cl_uint>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, &result));
        return result;
    }
};

template <class D> class ComplexArrayCL {
private:
    cl_mem data, reals, imaginaries;
    int size;
    bool planar;
public:
    ComplexArrayCL(cl::Context& context, ComplexArray<D>* array) : size(array->getSize()), planar(array->getPlanar()) {
        cl_int err;
        if (planar) {
            reals = clCreateBuffer(context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size*sizeof(D), array->getReals(), &err);
            xCLErr(err);
            imaginaries = clCreateBuffer(context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size*sizeof(D), array->getImaginaries(), &err);
            xCLErr(err);
        } else {
            data = clCreateBuffer(context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 2*size*sizeof(D), array->getData(), &err);
            xCLErr(err);
        }
    }

    ComplexArrayCL(cl::Context& context, int iSize, bool iPlanar) : size(iSize), planar(iPlanar) {
        cl_int err;
        if (planar) {
            reals = clCreateBuffer(context(), CL_MEM_READ_WRITE, size*sizeof(D), NULL, &err);
            xCLErr(err);
            imaginaries = clCreateBuffer(context(), CL_MEM_READ_WRITE, size*sizeof(D), NULL, &err);
            xCLErr(err);
        } else {
            data = clCreateBuffer(context(), CL_MEM_READ_WRITE, 2*size*sizeof(D), NULL, &err);
            xCLErr(err);
        }
    }

    ComplexArrayCL(cl::Context& context, PlannarizedComplexArray<GlobalPlannarLevel, D>* array) : size(array->getElements()/2), planar(false) {
        cl_int err;
        data = clCreateBuffer(context(), CL_MEM_READ_WRITE, array->getElements()*sizeof(D), NULL, &err);
        xCLErr(err);
    }

    cl_mem getReals() { return reals; }
    cl_mem getImaginaries() { return imaginaries; }
    cl_mem getData() { return data; }

    void enqueueReadArray(cl::CommandQueue& queue, ComplexArray<D>& a, bool blocking = true) {
        if (planar) {
            xCLErr(clEnqueueReadBuffer(queue(), reals, blocking ? CL_TRUE : CL_FALSE, 0, size*sizeof(D), a.getReals(), 0, NULL, NULL));
            xCLErr(clEnqueueReadBuffer(queue(), imaginaries, blocking ? CL_TRUE : CL_FALSE, 0, size*sizeof(D), a.getImaginaries(), 0, NULL, NULL));
        } else {
            xCLErr(clEnqueueReadBuffer(queue(), data, blocking ? CL_TRUE : CL_FALSE, 0, 2*size*sizeof(D), a.getData(), 0, NULL, NULL));
        }
    }

    void enqueueWriteArray(cl::CommandQueue& queue, ComplexArray<D>& a, bool blocking = false) {
        if (planar) {
            xCLErr(clEnqueueWriteBuffer(queue(), reals, blocking ? CL_TRUE : CL_FALSE, 0, size*sizeof(D), a.getReals(), 0, NULL, NULL));
            xCLErr(clEnqueueWriteBuffer(queue(), imaginaries, blocking ? CL_TRUE : CL_FALSE, 0, size*sizeof(D), a.getImaginaries(), 0, NULL, NULL));
        } else {
            xCLErr(clEnqueueWriteBuffer(queue(), data, blocking ? CL_TRUE : CL_FALSE, 0, 2*size*sizeof(D), a.getData(), 0, NULL, NULL));
        }
    }

    void enqueueReadArray(cl::CommandQueue& queue, PlannarizedComplexArray<GlobalPlannarLevel, D>& a, bool blocking = true) {
        xCLErr(clEnqueueReadBuffer(queue(), data, blocking ? CL_TRUE : CL_FALSE, 0, a.getElements()*sizeof(D), a.getData(), 0, NULL, NULL));
    }

    void enqueueWriteArray(cl::CommandQueue& queue, PlannarizedComplexArray<GlobalPlannarLevel, D>& a, bool blocking = false) {
        xCLErr(clEnqueueWriteBuffer(queue(), data, blocking ? CL_TRUE : CL_FALSE, 0, a.getElements()*sizeof(D), a.getData(), 0, NULL, NULL));
    }

    ~ComplexArrayCL() {
        if (planar) {
            clReleaseMemObject(reals);
            clReleaseMemObject(imaginaries);
        } else clReleaseMemObject(data);
    }
};

class OpenCLClass {
public:
};

/*class Access {
private:
    streng arrayName, indexExpression;
public:
    Access(streng iArrayName, streng iIndexExpression) : arrayName(iArrayName), indexExpression(iIndexExpression) { }
    operator streng () { return arrayName + streng("[") + indexExpression + streng("]"); }
};*/

class OpenCLAlgorithm {
protected:
    std::vector<cl::Device> devicesToUse;
    cl::Context* context;
    cl::CommandQueue* commandQueue;
public:
    Timer* timerComputation;
    std::vector<Timer> kernelTimers;

    OpenCLAlgorithm() : context(NULL), commandQueue(NULL), timerComputation(NULL) {
        std::vector<cl::Platform> platforms;
        xCLErr(cl::Platform::get(&platforms));

//        printf("Platforms: %i", (int) platforms.size()); fflush(stdout);
//        devicesToUse = new std::vector<cl::Device>();
        for (int p = 0; p < (int) platforms.size(); ++p) { CLPlatform platform = CLPlatform(platforms[p]);
//            printf("== Platform %i: %s ==", p, platform.name().c_str());
            /*printf("Vendor: %s", platform.vendor().c_str());
            printf("Profile: %s", platform.profile().c_str());
            printf("Version: %s", platform.version().c_str());
            printf("Extensions: %s", platform.extensions().c_str());
*/
            std::vector<cl::Device> devices;
//            xCLErr(platforms[p].getDevices(CL_DEVICE_TYPE_CPU, &devices));
            xCLErr(platforms[p].getDevices(CL_DEVICE_TYPE_GPU, &devices));
       //     qDe bug("%i devices", (int) devices.size());
            for (int d = 0; d < (int) devices.size(); ++d) { CLDevice device = CLDevice(devices[d]);
                printf("[%s %s]", device.vendor().c_str(), device.name().c_str());
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

        cl_int err;
// printf("Creating context..."); fflush(stdout);
        context = new cl::Context(devicesToUse, NULL, NULL, NULL, &err); xCLErr(err);
//printf("Creating command queue..."); fflush(stdout);
        commandQueue = new cl::CommandQueue(*context, devicesToUse[0], CL_QUEUE_PROFILING_ENABLE, &err); xCLErr(err);
    }

    virtual double getTotalComputationFlops(int kernel) { return 0; }
    virtual std::string getKernelInfo(int kernel) { return ""; }

    ~OpenCLAlgorithm() {
        kernelTimers.clear();
        delete context;
        delete commandQueue;
    }
};

#endif // UOPENCL_H
