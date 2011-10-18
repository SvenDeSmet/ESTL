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

#if defined(LINUX)
#include <sys/time.h>
#include <stdint.h>
#include <unistd.h>
#endif

#ifdef _WIN32
//#include <ctime>
 #include <windows.h>
 #include <winbase.h>
#endif

#ifdef MAC
#include <mach/mach.h>
#include <mach/mach_time.h>
#endif

#include "stdio.h"

class Timer {
private:

    double totalTime;
    bool started;
    int runs;


#ifdef MAC
    double lastStartTime;

    double getTime() { return mach_absolute_time(); }

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
#endif
#ifdef _WIN32
    LARGE_INTEGER lastStartTime;

    LARGE_INTEGER getTime() { LARGE_INTEGER result; QueryPerformanceCounter(&result); return result; }

    double subtractTimes(LARGE_INTEGER endTime, LARGE_INTEGER startTime) {
        LARGE_INTEGER ticksPerSecond;
        if (!QueryPerformanceFrequency(&ticksPerSecond)) printf("\tQueryPerformance not present");

        LARGE_INTEGER cputime;
        cputime.QuadPart = endTime.QuadPart - startTime.QuadPart;

        return ((double) cputime.QuadPart / (double) ticksPerSecond.QuadPart);
    }
#endif
#ifdef LINUX
    struct timeval lastStartTime;

    struct timeval getTime() { struct timeval result; gettimeofday(&result, NULL); return result; }

    double subtractTimes(struct timeval end, struct timeval start) {
        long seconds, useconds;

        seconds  = end.tv_sec  - start.tv_sec;
        useconds = end.tv_usec - start.tv_usec;

        return ((seconds) * 1000 + useconds/1000.0);
    }
#endif

public:
    Timer() : totalTime(0), started(false), runs(0) { }

    void resume() { lastStartTime = getTime(); started = true; }
    void suspend() { addRun(subtractTimes(getTime(), lastStartTime)); started = false; }

    void addRun(double runTime) { totalTime += runTime; runs++; }

    double getTotalTime() { return totalTime; }
    double getAverageTime() { return totalTime/runs; }
    int getRuns() { return runs; }
};

#endif // TIMER_H
