/* The information in this file is
 * Copyright (C) 2011, Sven De Smet <sven@cubiccarrot.com>
 * and is subject to the terms and conditions of the
 * GNU Lesser General Public License Version 2.1
 * The license text is available from
 * http://www.gnu.org/licenses/lgpl.html
 */

#ifndef FFT_MKL_H
#define FFT_MKL_H

#include "FFT.h"

#ifdef USE_MKL
template <class D> class FFT_MKL : public FFT<D> {
private:
    DFTI_DESCRIPTOR_HANDLE h;
public:
    FFT_MKL(int n) {
        in = new D[2*n];
        out = new D[2*n];

        DftiCreateDescriptor(&h, DFTI_SINGLE, DFTI_COMPLEX, 1, n);
        DftiSetValue(h, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiCommitDescriptor(h);
    }

    virtual void execute() { fftwf_execute(plan); }

    virtual ~FFT_MKL() {
        delete [] in;
        delete [] out;
        fftwf_destroy_plan(p1);
    }
};
#endif

#endif // FFT_MKL_H
