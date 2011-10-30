#QT       += core
QT       -= gui
CONFIG   += console
#CONFIG   -= app_bundle
#TEMPLATE = app


#MAC
mac:DEFINES += MAC
mac:LIBS += -lfftw3f -framework OpenCL
mac:DEFINES += USE_FFTW


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
    OpenCL_FFT/clFFT.h \
    tests/Timer.h \
    tests/Test.h \
    FFT_OpenCL_LAC.h \
    KernelGenerator.h \
    SimpleMath.h

SOURCES += \
    Complex.cpp \
    OpenCL_FFT/fft_setup.cpp \
    OpenCL_FFT/fft_kernelstring.cpp \
    OpenCL_FFT/fft_execute.cpp \
    tests/Timer.cpp \
    tests/Test.cpp \
    main.cpp \
    KernelGenerator.cpp

# LINUX/AMD:
linux:INCLUDEPATH += /opt/AMDAPP/include/
linux:LIBS += -L/opt/AMDAPP/lib/x86_64/
linux:LIBS += -L/opt/AMDAPP/lib/x86/
linux:LIBS += -lOpenCL

# WINDOWS/AMD
win32:INCLUDEPATH += "C:\Program Files (x86)\AMD APP\include"
win32:LIBS += -L$$quote(C:/Program Files (x86)/AMD APP/lib/x86/)
win32:LIBS += -L$$quote(C:/Program Files (x86)/AMD APP/lib/x86_64/)
win32:LIBS += -lOpenCL

# MAC
#DEFINES += __CL_ENABLE_EXCEPTIONS







