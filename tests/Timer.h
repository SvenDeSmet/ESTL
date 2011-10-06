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

#ifndef TIMER_H
#define TIMER_H

#include <sys/time.h>
#include <stdint.h>
#include <mach/mach.h>
#include <mach/mach_time.h>

class Timer {
private:
    double totalTime;
    double lastStartTime;
    bool started;
    int runs;

    double subtractTimes(uint64_t endTime, uint64_t startTime) {
        uint64_t difference = endTime - startTime;
        static double conversion = 0.0;

        if (conversion == 0.0) {
            mach_timebase_info_data_t info;
            kern_return_t err = mach_timebase_info( &info );

            //Convert the timebase into seconds
            if (err == 0) conversion = 1e-9 * (double) info.numer / (double) info.denom;
        }

        return conversion * (double) difference;
    }

public:
    Timer() : totalTime(0), lastStartTime(0), started(false), runs(0) { }

    void resume() { lastStartTime = mach_absolute_time(); started = true; }
    void suspend() { addRun(subtractTimes(mach_absolute_time(), lastStartTime)); started = false; }

    void addRun(double runTime) { totalTime += runTime; runs++; }

    double getTotalTime() { return totalTime; }
    double getAverageTime() { return totalTime/runs; }
    int getRuns() { return runs; }
};

#endif // TIMER_H
