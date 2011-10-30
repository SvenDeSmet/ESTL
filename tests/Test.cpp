/* The information in this file is
 * Copyright (C) 2011, Sven De Smet <sven@cubiccarrot.com>
 * and is subject to the terms and conditions of the
 * GNU Lesser General Public License Version 2.1
 * The license text is available from
 * http://www.gnu.org/licenses/lgpl.html
 */

#include "Test.h"

#include "../FFTs.h"

//#include <faad.h>
#include "Timer.h"
#include <exception>
#include <math.h>
#include "assert.h"

#define MAX_ERR 1E-3
#define ASSERT_APPROX_EQ_C(a, b) assert((a - b).getNormSquared() < MAX_ERR)
#define ASSERT_APPROX_EQ(a, b) assert(abs(a - b) < MAX_ERR)

class TestFailureException : public std::exception {
private:
    std::string msg;
public:
    TestFailureException(std::string iMsg) : msg(iMsg) {}

    virtual const char* what() const throw() {
        return (std::string("Test failed: ") + msg + std::string("\n")).c_str();
    }

    ~TestFailureException() throw() {}
};

template <class T> void assert_approx_equal(T a, T b, double factor = 1.0) {
    if (abs(a - b) > MAX_ERR*factor) throw TestFailureException("a != b");
}

template <class D> void assert_approx_equal(Complex<D> a, Complex<D> b, double factor = 1.0) {
    if ((a - b).getNormSquared() > MAX_ERR*factor) throw TestFailureException(a.toString() + std::string(" != ") + b.toString());
}

bool TestFourierFloat::execute() {
    typedef Complex<D> C;

    for (int b = 0; b < 23; ++b) { int B = 1 << b; // batch size
        for (int f = 0; f < (int) fftFactories.size(); ++f) {
            for (int m = 1; m < 24 - b; m++) { int M = 1 << m; // fft size
                if (B <= fftFactories[f]->getMaxBatchSize(M)) {
                    printf("[%i]@@@ %s @@@ m == %i: ", b, fftFactories[f]->getName().c_str(), m);
                    FFT<D>* fft = fftFactories[f]->newFFT(M, true, 1 << b);
                    Timer timer, timerComputation;

                    if (OpenCLAlgorithm* fft_opencl = dynamic_cast<OpenCLAlgorithm*>(fft)) { fft_opencl->timerComputation = &timerComputation; }

                    int maxComps = ((4096*1024*8)/M) >> b;
                    if (maxComps == 0) maxComps = 1;
                    for (int k = 1; (2*k < M) && (k < maxComps); k++) { //printf("k == %i ** (M = %i) ***********", k, M);
                        for (int bIx = 0; bIx < (1 << b); ++bIx) {
                            int effK = ((k + bIx) % (M/2 - 1)) + 1;
    //                        effK = k;
                            for (int q = 0; q < M; q++) fft->setData(q + bIx*M, (D) cos(2*M_PI*((1.0*effK*q)/M)));
                        }

                        timer.resume();
                        fft->execute();
                        timer.suspend();

                        for (int bIx = 0; bIx < (1 << b); ++bIx) {// printf("(bIx = %i)", bIx);
                            int effK = ((k + bIx) % (M/2 - 1)) + 1;
    //                        effK = k;
                            for (int q = 0; q < M; q++) {
                                C c =   (((effK > 0) && ((q == effK) || (q == (M - effK))) ? C((D) (M/2.0)) : C((D) 0.0))
                                      + ((effK == 0) && (q == 0)                     ? C((D) M)       : C((D) 0.0)));
                                assert_approx_equal(fft->getData(q + bIx*M), c, M);
                            }
                        }
                    }

                    double floatingOperations = (5.0*m*(1 << m)*B);
                    double GFlopsTotal = 1E-9*floatingOperations/timer.getAverageTime();
                    double GFlopsComputation = (timerComputation.getRuns() > 0) ? 1E-9*floatingOperations/timerComputation.getAverageTime() : -1;
                    if (GFlopsComputation >= 0) { printf(" %.2f Gfs (%.2f Gfs)\n", GFlopsTotal, GFlopsComputation); }
                    else { printf(" %.2f Gfs\n", GFlopsTotal); }
                    if (OpenCLAlgorithm* fft_opencl = dynamic_cast<OpenCLAlgorithm*>(fft)) if (fft_opencl->kernelTimers.size() > 0) { //printf("\n");
                        double totalT = 0;
                        for (int qG = 1; qG <= (int) fft_opencl->kernelTimers.size(); ++qG) {
                            double t = fft_opencl->kernelTimers[qG - 1].getAverageTime();
                            totalT += t;
                        }
                        double totalOps = 0;
                        for (int qG = 1; qG <= (int) fft_opencl->kernelTimers.size(); ++qG) {
                            double t = fft_opencl->kernelTimers[qG - 1].getAverageTime();
                            double ops = fft_opencl->getTotalComputationFlops(qG - 1);
                            totalOps += ops;
                            printf("[K%i (%s): %.2f Gfs (%i%%)]", qG, fft_opencl->getKernelInfo(qG - 1).c_str(), 1E-9*ops/t, (int) (100*(t/totalT) + 0.5));
                        }
                        printf("(Total: %.2f Gfs)", 1E-9*totalOps/totalT); printf("\n");
                    }
                    fflush(stdout);

                    delete fft;
                }
            }
        }
    }

    return true;
}

bool performTests() {
    bool result = true;

    typedef float T;
    std::vector<FFTFactory<T> *> factories;
   // factories.push_back(new FFTFactorySpecific<FFT_OpenCL_Contiguous_InnerKernelTester<T> >());
//    factories.push_back(new FFTFactorySpecific<FFT_FFTW3<T> >());
    factories.push_back(new FFTFactorySpecific<FFT_OpenCL_Contiguous<T> >());
    factories.push_back(new FFTFactorySpecific<FFT_OpenCL_A<T, clFFT_InterleavedComplexFormat, clFFT_OUT_OF_PLACE> >());
   factories.push_back(new FFTFactorySpecific<FFT_OpenCL_A<T, clFFT_SplitComplexFormat, clFFT_OUT_OF_PLACE> >());
    factories.push_back(new FFTFactorySpecific<FFT_OpenCL_A<T, clFFT_InterleavedComplexFormat, clFFT_IN_PLACE> >());
   factories.push_back(new FFTFactorySpecific<FFT_OpenCL_A<T, clFFT_SplitComplexFormat, clFFT_IN_PLACE> >());
//    factories.push_back(new FFTFactorySpecific<FFT_OpenCL_LAC<T> >());

    TestFourierFloat* test = new TestFourierFloat(factories);
    try { result = result && test->execute(); }
    catch (TestFailureException& e) { printf("%s", e.what()); }
    catch (std::exception& e) { printf("Unknown exception (%s)", e.what()); }

//    getchar();

    delete test;

    return result;
}




