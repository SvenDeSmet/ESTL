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



#ifndef FFT_H
#define FFT_H

#include "Complex.h"
#include <stdio.h>

template <class D> class DataInterface {
public:
    virtual inline void setElement(int index, Complex<D> value, int batchIndex = 0) = 0;
    virtual inline Complex<D> getElement(int index, int batchIndex = 0) = 0;

    virtual ~DataInterface() { };
};

template <class D> class PlannarizedDataInterface : public DataInterface<D> {
private:
    int size;
public:
    typedef PlannarizedComplexArray<GlobalPlannarLevel, D> Array;
    Array *in, *out;

    PlannarizedDataInterface(int iSize) : size(iSize) {
        in = new Array(size);
        out = new Array(size);
    }
    virtual ~PlannarizedDataInterface() { delete in; delete out; }

    inline void setElement(int index, Complex<D> value, int batchIndex = 0) { in->setElement(batchIndex*size + index, value); }
    inline Complex<D> getElement(int index, int batchIndex = 0) { return out->getElement(batchIndex*size + index); }
};

template <class D> class SplitInterleavedDataInterface : public DataInterface<D> {
private:
    int size;
public:
    typedef ComplexArray<D> Array;
    Array *in, *out;

    SplitInterleavedDataInterface(int iSize, bool plannar = true) : size(iSize) {
        in = new Array(size, plannar);
        out = new Array(size, plannar);
    }
    virtual ~SplitInterleavedDataInterface() { delete in; delete out; }

    inline void setElement(int index, Complex<D> value, int batchIndex = 0) { in->setElement(batchIndex*size + index, value); }
    inline Complex<D> getElement(int index, int batchIndex = 0) { return out->getElement(batchIndex*size + index); }
};

template <class D> class FFT {
protected:
public:
    typedef D DataType;

    DataInterface<D>* dataInterface;
    int size, batchCount;

    FFT(int iSize, int iBatchCount) : dataInterface(NULL), size(iSize), batchCount(iBatchCount) { }
    virtual ~FFT() { delete dataInterface; }

    inline void setData(int index, Complex<D> value, int batchIndex = 0) { dataInterface->setElement(batchIndex*size + index, value); }
    inline Complex<D> getData(int index, int batchIndex = 0) { return dataInterface->getElement(batchIndex*size + index); }

    virtual void execute() = 0;
};

template <class D> class FFTFactory {
public:
    virtual FFT<D>* newFFT(int n, bool inverse = false) = 0;
};

template <class FFTClass> class FFTFactorySpecific : public FFTFactory<typename FFTClass::DataType> {
    typedef typename FFTClass::DataType D;
public:
    virtual FFT<D>* newFFT(int n, bool inverse = false) { return (FFT<D>*) new FFTClass(n, inverse); }
};

template <class D> class FFT2D {
public:
    Complex<D> *data;
    int xDim, yDim;
    FFTFactory<D>* fftFactory;
    bool inverse;

    FFT2D(int iXDim, int iYDim, FFTFactory<D>* iFFTFactory, bool iInverse = false) : xDim(iXDim), yDim(iYDim), fftFactory(iFFTFactory), inverse(iInverse) {
        data = new Complex<D>[xDim*yDim];
    }

    virtual void execute() {
        FFT<D>* fftX = fftFactory->newFFT(xDim, inverse);
        for (int y = 0; y < yDim; ++y) { if ((y & 0x3FF) == 0) printf("%i:", y);
            for (int x = 0; x < xDim; ++x) fftX->setData(x, data[y*xDim + x]);
            fftX->execute();
            for (int x = 0; x < xDim; ++x) data[y*xDim + x] = fftX->getData(x);
        }
        delete fftX;

        FFT<D>* fftY = fftFactory->newFFT(yDim, inverse);
        for (int x = 0; x < xDim; ++x) {
            for (int y = 0; y < yDim; ++y) fftY->setData(y, data[y*xDim + x]);
            fftY->execute();
            for (int y = 0; y < yDim; ++y) data[y*xDim + x] = fftY->getData(y);
        }
        delete fftY;
    }

    ~FFT2D() { delete [] data; }
};

#endif // FFT_H
