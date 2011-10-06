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
