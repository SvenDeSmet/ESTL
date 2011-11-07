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
protected:
    int size;
public:
    A *in, *out;
    ArrayDataInterface(int iSize) : size(iSize) { }
    virtual ~ArrayDataInterface() { delete in; delete out; }

    inline void setElement(int index, Complex<D> value, int batchIndex = 0) { this->in->setElement(batchIndex*size + index, value); }
    inline Complex<D> getElement(int index, int batchIndex = 0) { return this->out->getElement(batchIndex*size + index); }
};

template <class D> class PlannarizedDataInterface : public ArrayDataInterface<D, PlannarizedComplexArray<GlobalPlannarLevel, D> > {
private:
public:
    typedef PlannarizedComplexArray<GlobalPlannarLevel, D> Array;

    PlannarizedDataInterface(int iSize) : ArrayDataInterface<D, PlannarizedComplexArray<GlobalPlannarLevel, D> >(iSize) {
        this->in = new Array(this->size);
        this->out = new Array(this->size);
    }
};

template <class D> class SplitInterleavedDataInterface : public ArrayDataInterface<D, ComplexArray<D> > {
public:
    typedef ComplexArray<D> Array;

    SplitInterleavedDataInterface(int iSize, bool plannar = true) : ArrayDataInterface<D, ComplexArray<D> >(iSize) {
        this->in = new Array(this->size, plannar);
        this->out = new Array(this->size, plannar);
    }
};

#endif // DATAINTERFACE_H
