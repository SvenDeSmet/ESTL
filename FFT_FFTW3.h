/* The information in this file is
 * Copyright (C) 2011, Sven De Smet <sven@cubiccarrot.com>
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
    SplitInterleavedDataInterface<D>* splitInterleavedDataInterface;
public:

    FFT_FFTW3(int n, bool inverse = false, int iBatchCount = 1) : FFT<D>(n, iBatchCount) {
       this->dataInterface = splitInterleavedDataInterface = new SplitInterleavedDataInterface<D>(this->batchCount*this->size, false);

        plan = fftwf_plan_many_dft(1, &this->size, this->batchCount, (fftwf_complex *) splitInterleavedDataInterface->in->getData(), NULL, 1, this->size,
                                                        (fftwf_complex *) splitInterleavedDataInterface->out->getData(), NULL, 1, this->size,
                                                        !inverse ? FFTW_FORWARD : FFTW_BACKWARD, FFTW_ESTIMATE);
    }

    virtual void execute() { fftwf_execute(plan); }

    virtual ~FFT_FFTW3() { fftwf_destroy_plan(plan); }
};

#endif // FFT_FFTW3_H
