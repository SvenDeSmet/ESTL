OTHER_FILES += \
    README \
    OpenCL_FFT/ReadMe.txt \
    OpenCL_FFT/param.txt \
    Measurements.txt

HEADERS += \
    uOpenCL.h \
    FFTs.h \
    FFT.h \
    FFT_OpenCL_Contiguous.h \
    FFT_OpenCL_Contiguous_InnerKernelTester.h \
    FFT_OpenCL_A.h \
    FFT_MKL.h \
    FFT_FFTW3.h \
    Convolution.h \
    ContiguousKernelGenerator.h \
    Complex.h \
    cl.hpp \
    OpenCL_FFT/procs.h \
    OpenCL_FFT/fft_internal.h \
    OpenCL_FFT/fft_base_kernels.h \
    OpenCL_FFT/clFFT.h \
    tests/Timer.h \
    tests/Test.h

SOURCES += \
    Complex.cpp \
    OpenCL_FFT/fft_setup.cpp \
    OpenCL_FFT/fft_kernelstring.cpp \
    OpenCL_FFT/fft_execute.cpp \
    tests/Timer.cpp \
    tests/Test.cpp \
    main.cpp \
    ContiguousKernelGenerator.cpp


LIBS += -lfftw3f -framework OpenCL










