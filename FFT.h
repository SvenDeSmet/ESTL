/* The information in this file is
 * Copyright (C) 2011, Sven De Smet <sven@cubiccarrot.com>
 * and is subject to the terms and conditions of the
 * GNU Lesser General Public License Version 2.1
 * The license text is available from
 * http://www.gnu.org/licenses/lgpl.html
 */

#ifndef FFT_H
#define FFT_H

#include "Complex.h"
#include <stdio.h>
#include <string>
#include <DataInterface.h>

typedef std::string streng;

template <class D> class FFT {
protected:
public:
    typedef D DataType;

    DataInterface<D>* dataInterface;
    int size, batchCount;

    FFT(int iSize, int iBatchCount) : dataInterface(NULL), size(iSize), batchCount(iBatchCount) { }
    virtual ~FFT() { delete dataInterface; }

    static int getMaxBatchSize(int n) { return 1; }

    inline void setData(int index, Complex<D> value, int batchIndex = 0) { dataInterface->setElement(batchIndex*size + index, value); }
    inline Complex<D> getData(int index, int batchIndex = 0) { return dataInterface->getElement(batchIndex*size + index); }

    virtual void execute() = 0;
};

template <class D> class FFTFactory {
public:
    virtual streng getName() = 0;
    virtual int getVersion() { return 0; }
    virtual FFT<D>* newFFT(int n, bool forward = true, int batchSize = 1) = 0;
    virtual int getMaxBatchSize(int n) { return 1; }
};

template <class FFTClass> class FFTFactorySpecific : public FFTFactory<typename FFTClass::DataType> {
    typedef typename FFTClass::DataType D;
public:
    virtual streng getName() { return FFTClass::getName(); }
    virtual int getMaxBatchSize(int n) { return FFTClass::getMaxBatchSize(n); }
    virtual FFT<D>* newFFT(int n, bool forward = true, int batchSize = 1) { return (FFT<D>*) new FFTClass(n, forward, batchSize); }
};

template <class D> class FFT2D {
public:
    Complex<D> *data;
    int xDim, yDim;
    FFTFactory<D>* fftFactory;
    bool forward;

    FFT2D(int iXDim, int iYDim, FFTFactory<D>* iFFTFactory, bool iForward = true) : xDim(iXDim), yDim(iYDim), fftFactory(iFFTFactory), forward(iForward) {
        data = new Complex<D>[xDim*yDim];
    }

    virtual void execute() {
        FFT<D>* fftX = fftFactory->newFFT(xDim, forward);
        for (int y = 0; y < yDim; ++y) {// if ((y & 0x3FF) == 0) printf("%i:", y);
            for (int x = 0; x < xDim; ++x) fftX->setData(x, data[y*xDim + x]);
            fftX->execute();
            for (int x = 0; x < xDim; ++x) data[y*xDim + x] = fftX->getData(x);
        }
        delete fftX;

        FFT<D>* fftY = fftFactory->newFFT(yDim, forward);
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
