#include "Test.h"

#include "../FFTs.h"

//#include <faad.h>
#include "Timer.h"

#include <math.h>
#include "assert.h"

#define MAX_ERR 1E-1
#define ASSERT_APPROX_EQ_C(a, b) assert((a - b).getNormSquared() < MAX_ERR)
#define ASSERT_APPROX_EQ(a, b) assert(abs(a - b) < MAX_ERR)


bool TestFourierFloat::execute() {
    typedef Complex<D> C;

    for (int f = 0; f < fftFactories.size(); ++f) {
        for (int m = 0; m < 22; m++) { int M = 1 << m; //qDebug("%i", m);
            printf("@@@ m == %i: ", m);
            FFT<D>* fft = fftFactories[f]->newFFT(M);
            Timer timer, timerComputation;

            if (FFT_OpenCL_Contiguous<D>* fft_opencl = dynamic_cast<FFT_OpenCL_Contiguous<D>*>(fft)) { fft_opencl->timerComputation = &timerComputation; }

            int maxComps = 4096/(1 << m);
            if (maxComps == 0) maxComps = 1;

            for (int k = 0; (2*k < M) && (k < maxComps); k++) {// qDebug("k == %i ** (M = %i) ***********", k, M);
                for (int q = 0; q < M; q++) fft->setData(q, cos(2*M_PI*((1.0 * k*q)/M)));

                timer.resume();
                fft->execute();
                timer.suspend();

                for (int q = 0; q < M; q++) {
                    C c = ((k > 0) && ((q == k) || (q == (M - k))) ? C(M/2.0) : C(0.0)) + ((k == 0) && (q == 0) ? C(M) : C(0.0));
                    if (!((fft->getData(q) - c).getNormSquared() < MAX_ERR)) {
                        printf("%i: ", q); c.print(); fft->getData(q).print();
                    }//toString().c_str());// series[q].print();
                    ASSERT_APPROX_EQ_C(fft->getData(q), c);
                }
            }

            //char temp[100];
            int floatingOperations = (5.0*m*(1 << m));
            double GFlopsTotal = 1E-9*floatingOperations/timer.getAverageTime();
            double GFlopsComputation = (timerComputation.getRuns() > 0) ? 1E-9*floatingOperations/timerComputation.getAverageTime() : -1;
            printf(" %f GFlops/s (%f GFlops/s)\n", GFlopsTotal, GFlopsComputation);

            delete fft;
        }
    }

    return true;
}


int performTests() {
    bool result = true;

    typedef float T;
    std::vector<FFTFactory<T> *> factories;
   // factories.push_back(new FFTFactorySpecific<FFT_OpenCL_Contiguous_InnerKernelTester<T> >());
   // factories.push_back(new FFTFactorySpecific<FFT_FFTW3<T> >());
    factories.push_back(new FFTFactorySpecific<FFT_OpenCL_Contiguous<T> >());
    factories.push_back(new FFTFactorySpecific<FFT_OpenCL_A<T> >());

    TestFourierFloat* test = new TestFourierFloat(factories);
    result = result && test->execute();
    delete test;

    return result;
}




