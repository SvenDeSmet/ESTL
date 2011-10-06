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

//        plan = fftwf_plan_dft_1d(n, (fftwf_complex *) this->in->getData(), (fftwf_complex *) this->out->getData(), !inverse ? FFTW_FORWARD : FFTW_BACKWARD, FFTW_ESTIMATE);
        plan = fftwf_plan_many_dft(1, &this->size, this->batchCount, (fftwf_complex *) splitInterleavedDataInterface->in->getData(), NULL, 1, this->size,
                                                        (fftwf_complex *) splitInterleavedDataInterface->out->getData(), NULL, 1, this->size,
                                                        !inverse ? FFTW_FORWARD : FFTW_BACKWARD, FFTW_ESTIMATE);
    }

    virtual void execute() { fftwf_execute(plan); }

    virtual ~FFT_FFTW3() { fftwf_destroy_plan(plan); }
};

#endif // FFT_FFTW3_H
