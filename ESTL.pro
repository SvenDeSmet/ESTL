
# LINUX/AMD:
#INCLUDEPATH += /opt/AMDAPP/include/
#DEFINES += LINUX
#LIBS += /opt/AMDAPP/lib/x86_64/libOpenCL.so

# WINDOWS/AMD
#INCLUDEPATH += "C:\Program Files (x86)\AMD APP\include"
#?? DEFINES += _WIN32
#?? DEFINES += CL_CALLBACK __stdcall
#LIBS += $$quote(C:/Program Files (x86)/AMD APP/lib/x86_64/libOpenCL.a)
#LIBS += -L$$quote(C:/Program Files (x86)/AMD APP/lib/x86_64/) -lOpenCL

# MAC
DEFINES += MAC
LIBS += -lfftw3f -framework OpenCL

#DEFINES += __CL_ENABLE_EXCEPTIONS

OTHER_FILES += \
    README \
    OpenCL_FFT/ReadMe.txt \
    OpenCL_FFT/param.txt \
    Measurements.txt \
    Issues.txt

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
    Complex.h \
    cl.hpp \
    OpenCL_FFT/procs.h \
    OpenCL_FFT/fft_internal.h \
    OpenCL_FFT/fft_base_kernels.h \
    OpenCL_FFT/clFFT.h \
    tests/Timer.h \
    tests/Test.h \
    FFT_OpenCL_LAC.h \
    KernelGenerator.h \
    SimpleMath.h \
    TestCode.h

SOURCES += \
    Complex.cpp \
    OpenCL_FFT/fft_setup.cpp \
    OpenCL_FFT/fft_kernelstring.cpp \
    OpenCL_FFT/fft_execute.cpp \
    tests/Timer.cpp \
    tests/Test.cpp \
    main.cpp \
    ContiguousKernelGenerator.cpp










