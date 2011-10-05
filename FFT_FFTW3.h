/*
 * The information in this file is
 * Copyright (C) 2010-2011, Sven De Smet <sven@cubiccarrot.com>
 * and is subject to the terms and conditions of the
 * GNU Lesser General Public License Version 2.1
 * The license text is available from
 * http://www.gnu.org/licenses/lgpl.html
 */

#ifndef FFT_FFTW3_H
#define FFT_FFTW3_H

#include "FFT.h"
#include <fftw3.h>

template <class D> class FFT_FFTW3 : public FFT<D> {
private:
    fftwf_plan plan;
public:
    FFT_FFTW3(int n, bool inverse = false, int iBatchCount = 1) : FFT<D>(n, iBatchCount) {
        this->in = new ComplexArray<D>(this->batchCount*this->size, false);
        this->out = new ComplexArray<D>(this->batchCount*this->size, false);

//        plan = fftwf_plan_dft_1d(n, (fftwf_complex *) this->in->getData(), (fftwf_complex *) this->out->getData(), !inverse ? FFTW_FORWARD : FFTW_BACKWARD, FFTW_ESTIMATE);
        plan = fftwf_plan_many_dft(1, &this->size, this->batchCount, (fftwf_complex *) this->in->getData(), NULL, 1, this->size,
                                                        (fftwf_complex *) this->out->getData(), NULL, 1, this->size,
                                                        !inverse ? FFTW_FORWARD : FFTW_BACKWARD, FFTW_ESTIMATE);
    }

    virtual void execute() { fftwf_execute(plan); }

    ~FFT_FFTW3() {
        fftwf_destroy_plan(plan);

        delete this->in;
        delete this->out;
    }
};

#endif // FFT_FFTW3_H
