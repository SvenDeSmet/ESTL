#ifndef TEST_H_
#define TEST_H_

#include "../FFT.h"
#include <vector>

class Test {
public:
    virtual bool execute() = 0;
};

class TestFourierFloat : public Test {
private:
    typedef float D;
    std::vector<FFTFactory<D> *> fftFactories;
public:
    TestFourierFloat(std::vector<FFTFactory<D> *> iFFTFactories) : fftFactories(iFFTFactories) { }

    bool execute();
};

int performTests();

#endif /* TEST_H_ */
