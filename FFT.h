/*
 * The information in this file is
 * Copyright (C) 2010-2011, Sven De Smet <sven@cubiccarrot.com>
 * and is subject to the terms and conditions of the
 * GNU Lesser General Public License Version 2.1
 * The license text is available from
 * http://www.gnu.org/licenses/lgpl.html
 */


#ifndef FFT_H
#define FFT_H

#include "Complex.h"
#include <stdio.h>

template <class D> class FFT {
public:
    typedef D DataType;

    int size;
    int batchCount;
    ComplexArray<D> *in, *out;

    void setData(int index, Complex<D> value, int batchIndex = 0) { in->setElement(batchIndex*size + index, value); }
    Complex<D> getData(int index, int batchIndex = 0) { return out->getElement(batchIndex*size + index); }

    virtual void execute() = 0;

    FFT(int iSize, int iBatchCount) : size(iSize), batchCount(iBatchCount) { }

    virtual ~FFT() { }
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
