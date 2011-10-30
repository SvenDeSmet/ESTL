/* The information in this file is
 * Copyright (C) 2011, Sven De Smet <sven@cubiccarrot.com>
 * and is subject to the terms and conditions of the
 * GNU Lesser General Public License Version 2.1
 * The license text is available from
 * http://www.gnu.org/licenses/lgpl.html
 */

#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "Complex.h"
#include "FFTs.h"

template <class T>
class Data2D {
public:
    virtual int getWidth() = 0;
    virtual int getHeight() = 0;
    virtual T getPixel(int x, int y) = 0;
    virtual void setPixel(int x, int y, T value) = 0;
};

template <class T>
class Convolution2D {
public:
    virtual void convolve(bool correlate = false) = 0;
};

template <class T>
class Convolution2D_Direct : public Convolution2D<T> {
private:
    Data2D<T> *dataToConvolve, *convolutionKernel, *result;
public:
    bool cyclic[2];

    Convolution2D_Direct(Data2D<T>* iDataToConvolve, Data2D<T>* iConvolutionKernel, Data2D<T>* iResult)
    : dataToConvolve(iDataToConvolve), convolutionKernel(iConvolutionKernel), result(iResult)
    { cyclic[0] = cyclic[1] = false; }

    virtual void convolve(bool correlate = false) {
        double weight = 0;
        for (int r = 0; r < convolutionKernel->getHeight(); ++r) {
            for (int q = 0; q < convolutionKernel->getWidth(); ++q) {
                weight += convolutionKernel->getPixel(q, r);
            }
        }
       // qDebug("weight = %f", weight);

        int xOffs = (convolutionKernel->getWidth() + 1)/2;
        int yOffs = (convolutionKernel->getHeight() + 1)/2;

        for (int y = yOffs; y < dataToConvolve->getHeight() - (convolutionKernel->getHeight() - yOffs); ++y) { if ((y & 0xF) == 0) qDebug("%i", y);
            for (int x = xOffs; x < dataToConvolve->getWidth() - (convolutionKernel->getWidth() - xOffs); ++x) {
                double acc = 0;
                if (correlate) {
                    for (int r = 0; r < convolutionKernel->getHeight(); ++r) {
                        for (int q = 0; q < convolutionKernel->getWidth(); ++q) {
                            acc += convolutionKernel->getPixel(q, r) * dataToConvolve->getPixel(x - xOffs + q, y - yOffs + r);
                        }
                    }
                } else { // convolve
                    for (int r = 0; r < convolutionKernel->getHeight(); ++r) {
                        for (int q = 0; q < convolutionKernel->getWidth(); ++q) {
                            acc += convolutionKernel->getPixel(convolutionKernel->getWidth() - q, convolutionKernel->getHeight() - r) * dataToConvolve->getPixel(x - xOffs + q, y - yOffs + r);
                        }
                    }
                }
                if (weight != 0) acc /= (weight);
                result->setPixel(x, y, (int) (acc + 0.5));
            }
        }
    }
};

#define m_a_x(a, b) ((a) > (b) ? (a) : (b))

template <class T>
class Convolution2D_FFT : public Convolution2D<T> {
private:
    Data2D<T> *dataToConvolve, *convolutionKernel, *result;
public:
    bool cyclic[2];

    Convolution2D_FFT(Data2D<T>* iDataToConvolve, Data2D<T>* iConvolutionKernel, Data2D<T>* iResult)
    : dataToConvolve(iDataToConvolve), convolutionKernel(iConvolutionKernel), result(iResult)
    { cyclic[0] = cyclic[1] = false; }

    int nextPowerOf2(int n) {
        int result = 1;
        while (result < n) result *= 2;
        return result;
    }

    virtual void convolve(bool correlate = false) {
        int width = nextPowerOf2(dataToConvolve->getWidth() + convolutionKernel->getWidth() - 1);
        int height = nextPowerOf2(dataToConvolve->getHeight() + convolutionKernel->getHeight() - 1);

        //FFTFactory<T>* fftFactory = new FFTFactorySpecific<FFT_FFTW3<T> >();
        typedef float S;
        FFTFactory<S>* fftFactory = new FFTFactorySpecific<FFT_OpenCL_Contiguous<S> >();
        FFT2D<S>* fft2DData = new FFT2D<S>(width, height, fftFactory, true);
        FFT2D<S>* fft2DConvolutionKernel = new FFT2D<S>(width, height, fftFactory, true);
        FFT2D<S>* fft2DDataInv = new FFT2D<S>(width, height, fftFactory, false);

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                bool extraData = (x >= dataToConvolve->getWidth()) || (y >= dataToConvolve->getHeight());
                fft2DData->data[y*width + x] = extraData ? 0 : dataToConvolve->getPixel(x, y);
            }
        }

        int xOffs = (convolutionKernel->getWidth() + 1)/2;
        int yOffs = (convolutionKernel->getHeight() + 1)/2;
        for (int y = 0; y < height; ++y) { int sourceY = (y + yOffs) % height;
            for (int x = 0; x < width; ++x) { int sourceX = (x + xOffs) % width;
                bool extraData = (sourceX >= convolutionKernel->getWidth()) || (sourceY >= convolutionKernel->getHeight());
                fft2DConvolutionKernel->data[y*width + x] = extraData ? 0 : convolutionKernel->getPixel(sourceX, sourceY);
            }
        }

        fft2DData->execute();
        fft2DConvolutionKernel->execute();

        if (correlate) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    fft2DDataInv->data[y*width + x] = fft2DData->data[y*width + x] * fft2DConvolutionKernel->data[y*width + x].getConjugate();
                }
            }
        } else { // convolve
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    fft2DDataInv->data[y*width + x] = fft2DData->data[y*width + x] * fft2DConvolutionKernel->data[y*width + x];
                }
            }
        }

        fft2DDataInv->execute();

        double scaleF = 1.0/fft2DConvolutionKernel->data[0].getReal();
        //qDebug("scaleF = %f", (float) fft2DConvolutionKernel->data[0].getReal());
        scaleF /= (((double) width)*height);
        //qDebug("scaleF = %f", (float) scaleF);

        for (int y = 0; y < dataToConvolve->getHeight(); ++y) {
            for (int x = 0; x < dataToConvolve->getWidth(); ++x) result->setPixel(x, y, scaleF*fft2DDataInv->data[y*width + x].getReal());
        }

        delete fft2DData;
        delete fft2DDataInv;
        delete fft2DConvolutionKernel;
        delete fftFactory;
    }
};

#endif // CONVOLUTION_H
