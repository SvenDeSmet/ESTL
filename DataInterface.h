#ifndef DATAINTERFACE_H
#define DATAINTERFACE_H

#include <Complex.h>

template <class D> class DataInterface {
public:
    virtual inline void setElement(int index, Complex<D> value, int batchIndex = 0) = 0;
    virtual inline Complex<D> getElement(int index, int batchIndex = 0) = 0;

    virtual ~DataInterface() { }
};

template <class D, class A> class ArrayDataInterface : public DataInterface<D> {
public:
    A *in, *out;
    virtual ~ArrayDataInterface() { delete in; delete out; }
};

template <class D> class PlannarizedDataInterface : public ArrayDataInterface<D, PlannarizedComplexArray<GlobalPlannarLevel, D> > {
private:
    int size;
public:
    typedef PlannarizedComplexArray<GlobalPlannarLevel, D> Array;

    PlannarizedDataInterface(int iSize) : size(iSize) {
        this->in = new Array(size);
        this->out = new Array(size);
    }

    inline void setElement(int index, Complex<D> value, int batchIndex = 0) { this->in->setElement(batchIndex*size + index, value); }
    inline Complex<D> getElement(int index, int batchIndex = 0) { return this->out->getElement(batchIndex*size + index); }
};

template <class D> class SplitInterleavedDataInterface : public ArrayDataInterface<D, ComplexArray<D> > {
private:
    int size;
public:
    typedef ComplexArray<D> Array;

    SplitInterleavedDataInterface(int iSize, bool plannar = true) : size(iSize) {
        this->in = new Array(size, plannar);
        this->out = new Array(size, plannar);
    }

    inline void setElement(int index, Complex<D> value, int batchIndex = 0) { this->in->setElement(batchIndex*size + index, value); }
    inline Complex<D> getElement(int index, int batchIndex = 0) { return this->out->getElement(batchIndex*size + index); }
};

#endif // DATAINTERFACE_H
