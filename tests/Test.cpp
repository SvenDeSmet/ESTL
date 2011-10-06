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

template <class T>
void assert_approx_equal(T a, T b, double factor = 1.0) {
    if (abs(a - b) > MAX_ERR*factor) throw TestFailureException("a != b");
}


template <class D>
void assert_approx_equal(Complex<D> a, Complex<D> b, double factor = 1.0) {
    if ((a - b).getNormSquared() > MAX_ERR*factor) throw TestFailureException(a.toString() + std::string(" != ") + b.toString());
}

bool TestFourierFloat::execute() {
    typedef Complex<D> C;

    for (int f = 0; f < fftFactories.size(); ++f) {
        for (int m = 1; m < 22; m++) { int M = 1 << m; //qDebug("%i", m);
            printf("@@@ m == %i: ", m);
            FFT<D>* fft = fftFactories[f]->newFFT(M);
            Timer timer, timerComputation;

            if (FFT_OpenCL_Contiguous<D>* fft_opencl = dynamic_cast<FFT_OpenCL_Contiguous<D>*>(fft)) { fft_opencl->timerComputation = &timerComputation; }

            int maxComps = (4096*1024*8)/(1 << m);
            if (maxComps == 0) maxComps = 1;

            for (int k = 0; (2*k < M) && (k < maxComps); k++) {// qDebug("k == %i ** (M = %i) ***********", k, M);
                for (int q = 0; q < M; q++) fft->setData(q, cos(2*M_PI*((1.0 * k*q)/M)));

                timer.resume();
                fft->execute();
                timer.suspend();

                for (int q = 0; q < M; q++) {
                    C c = ((k > 0) && ((q == k) || (q == (M - k))) ? C(M/2.0) : C(0.0)) + ((k == 0) && (q == 0) ? C(M) : C(0.0));
                    assert_approx_equal(fft->getData(q), c, M);
                }
            }

            //char temp[100];
            int floatingOperations = (5.0*m*(1 << m));
            double GFlopsTotal = 1E-9*floatingOperations/timer.getAverageTime();
            double GFlopsComputation = (timerComputation.getRuns() > 0) ? 1E-9*floatingOperations/timerComputation.getAverageTime() : -1;
            printf(" %f GFlops/s (%f GFlops/s)\n", GFlopsTotal, GFlopsComputation);
            fflush(stdout);

            delete fft;
        }
    }

    return true;
}

bool performTests() {
    bool result = true;

            printf("Starting tests");  fflush(stdout);

    typedef float T;
    std::vector<FFTFactory<T> *> factories;
   // factories.push_back(new FFTFactorySpecific<FFT_OpenCL_Contiguous_InnerKernelTester<T> >());
   // factories.push_back(new FFTFactorySpecific<FFT_FFTW3<T> >());
    factories.push_back(new FFTFactorySpecific<FFT_OpenCL_Contiguous<T> >());
   // factories.push_back(new FFTFactorySpecific<FFT_OpenCL_A<T> >());

    TestFourierFloat* test = new TestFourierFloat(factories);
    try { result = result && test->execute(); }
    catch (TestFailureException& e) { printf(e.what()); }
    catch (std::exception& e) { printf("Unknown exception (%s)", e.what()); }

    delete test;

    return result;
}




