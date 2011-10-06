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


#ifndef CCOMPLEX_H_
#define CCOMPLEX_H_

#include <stdlib.h>
#include <math.h>
#include <string>

#define GlobalAlignLevel 5
template <class D> class AlignedArray {
private:
        unsigned char* mem;
	
public:
        D* alignedData;
	
	AlignedArray(int size, int alignLevel = GlobalAlignLevel) {
	    mem = new unsigned char[size * sizeof(D) + ((1 << alignLevel) - 1)];
            alignedData = (D*) ((((long int) mem + ((1 << alignLevel) - 1)) >> alignLevel) << alignLevel);
	}
	
        inline D& operator [] (int index) { return alignedData[index]; }
        inline const D operator [] (int index) const { return alignedData[index]; }
	
        ~AlignedArray() { delete [] mem; }
};

template <int plannarLevel, int elementLevel, class Array, class D>
class ArrayPlannarizer {
private:
    static const int plannarMask = (1 << plannarLevel) - 1;
    Array& array;
public:
    ArrayPlannarizer(Array& iArray) : array(iArray) {}

    inline int getPosition(int index, int element) const {
        return ((index >> plannarLevel) << (plannarLevel + elementLevel)) | (element << plannarLevel) | (index & plannarMask);
    }

    inline void setElement(int index, int element, const D value) { array[getPosition(index, element)] = value; }
    inline const D getElement(int index, int element) const { return array[getPosition(index, element)]; }
};

#define ScaleFact 16
template <class R>
class Complex {
	template <class S> friend Complex<S> operator + (const Complex<S> a, const Complex<S> b);
	template <class S> friend Complex<S> operator - (const Complex<S> a, const Complex<S> b);
	template <class S> friend Complex<S> operator * (const Complex<S> a, const Complex<S> b);
	template <class S> friend Complex<S> operator >> (const Complex<S> a, int shift);
	template <class S> friend Complex<S> operator * (const S a, const Complex<S> b);
	template <class S> friend Complex<S> operator / (const Complex<S> b, const S a);
	template <class S> friend bool operator == (const Complex<S> a, const Complex<S> b);
private:
	R r, i;
public:
        Complex<R>(R iR, R iI) : r(iR), i(iI) { }
        Complex<R>() { }
        Complex<R>(R iR) : r(iR), i((R) 0.0) { }
        static Complex<R> unit(double iPhase) { return Complex<R>(cos(iPhase), sin(iPhase)); }
        void print() const { printf("(%f, %f)\n", r, i); }
        std::string toString() const { char result[128]; sprintf(result, "(%f, %f)", r, i); return result; }
        R getNorm() const { return sqrt(getNormSquared()); }
        R getNormSquared() const { return ((*this) * ((*this).getConjugate())).getReal(); }
        Complex<R> getNormalizedComplex() const { return (*this)/((R) sqrt(getNormSquared())); }
        Complex<R> getConjugate() const { return Complex<R>(r, -i); }
	
        inline R getReal() const { return r; }
        inline void setReal(R value) { r = value; }
        inline R getImaginary() const { return i; }
        inline void setImaginary(R value) { i = value; }
        double getAngle() const {
		Complex<R> n = getNormalizedComplex();
		return atan2(n.i, n.r); 
        }
};

template <class R> Complex<R> inline operator + (const Complex<R> a, const Complex<R> b) { return Complex<R>(a.r + b.r, a.i + b.i); }
template <class R> Complex<R> inline operator - (const Complex<R> a, const Complex<R> b) { return Complex<R>(a.r - b.r, a.i - b.i); }
template <class R> bool operator == (const Complex<R> a, const Complex<R> b) { return (a.r == b.r) && (a.i == b.i); }
template <class R> Complex<R> inline operator >> (const Complex<R> a, int shift) { return ((R) pow(0.5, shift)) * a; }

template <class R, class U, int L> R mul(R a, R b) { return (((U) a) * ((U) b)) >> L; }

template <class R> Complex<R> inline operator * (const R c, const Complex<R> b) { return Complex<R>(c * b.r, c * b.i); }
template <class R> Complex<R> inline operator * (const Complex<R> a, const Complex<R> b) {
	return Complex<R>(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r);
}

#define GlobalPlannarLevel 5
template <int plannarLevel, class D>
class PlannarizedComplexArray {
private:
    typedef ArrayPlannarizer<plannarLevel, 1, AlignedArray<D>, D> PlannarizedArray;
    PlannarizedArray* plannarizedData;
    AlignedArray<D>* data;
    int size;
    int elements;
public:
    PlannarizedComplexArray(int iSize) : size(iSize) {
        int roundLevel = plannarLevel + 1;
        elements = 2*size;
        elements = roundLevel*((elements + ((1 << roundLevel) - 1))/roundLevel);
        data = new AlignedArray<D>(elements);
        plannarizedData = new PlannarizedArray(*data);
    }

    int getElements() const { return elements; }

    void setElement(int index, Complex<D> value) {
        plannarizedData->setElement(index, 0, value.getReal());
        plannarizedData->setElement(index, 1, value.getImaginary());
    }

    Complex<D> getElement(int index) {
        return Complex<D>(plannarizedData->getElement(index, 0), plannarizedData->getElement(index, 1));
    }

    D* getData() { return &(*data)[0]; }

    int getSize() const { return size; }

    ~PlannarizedComplexArray() { delete plannarizedData; delete data; }
};

// NOTE: Integer specialization probably does not work correctly...
short int inline imul16(const short int a, const short int b) {  return mul<short int, int, 16>(a, b); }

template <class R> R isub(R a, R b) { return ((a >> 1) - (b >> 1)); }
template <class R> R iadd(R a, R b) { return ((a >> 1) + (b >> 1)); }

short int inline isub16(const short int a, const short int b) { return isub<short int>(a, b); }
short int inline iadd16(const short int a, const short int b) { return iadd<short int>(a, b); }
short int inline isub32(const short int a, const short int b) { return isub<short int>(a, b); }
short int inline iadd32(const int a, const short int b) { return iadd<short int>(a, b); }

int inline imul16_32(const short int a, const short int b) { return mul<short int, int, 16>(a, b); }

template <> Complex<short int> inline operator * (const short int c, const Complex<short int> b) { 
	return Complex<short int>(imul16(c, b.r), imul16(c, b.i)); 
}

template <> Complex<short int> inline operator * (const Complex<short int> a, const Complex<short int> b) {
	return Complex<short int>((((int) a.r * (int) b.r - (int) a.i * (int) b.i)) >> 14, 
		                      (((int) a.r * (int) b.i + (int) a.i * (int) b.r)) >> 14);
}


template <class R> Complex<R> inline operator / (const Complex<R> b, const R c) { return (1/c) * b; }




template <class D> class ComplexArray {
private:
    AlignedArray<D>* data;
    int size;
    bool planar;
    int planarGroupSize;
public:
    ComplexArray(int iSize, bool iPlanar = true) : size(iSize), planar(iPlanar) {
        data = new AlignedArray<D>(2*size);
        planarGroupSize = planar ? size : 1;
    }

    D* getReals() { return &(*data)[0]; }
    D* getImaginaries() { return planar ? &(*data)[size] : &(*data)[1]; }

    D* getData() { return &(*data)[0]; }

    void setElement(int index, Complex<D> value) {
        if (planar) {
            (*data)[index] = value.getReal();
            (*data)[planarGroupSize + index] = value.getImaginary();
        } else {
            (*data)[2*index] = value.getReal();
            (*data)[2*index + 1] = value.getImaginary();
        }
    }

    Complex<D> getElement(int index) {
        if (planar) { return Complex<D>((*data)[index], (*data)[planarGroupSize + index]); }
        else { return Complex<D>((*data)[2*index], (*data)[2*index + 1]); }
    }

    int getSize() const { return size; }
    bool getPlanar() const { return planar; }

    ~ComplexArray() { delete data; }
};
/*
template <class D> class ComplexArrayAbstract {
private:
    AlignedArray<D>* data;
    int size;
    bool planar;
    int planarGroupSize;
public:
    class ComplexAbstract {
    private:
        const int element;
    public:
        ComplexAbstract(int iElement) : element(iElement) { }

        D& r() { return planar ? data[]; }
        D& i() { return planar ? data[]; }
    };

    ComplexAbstract operator [] (int index) { return ComplexAbstract(index); }

    ComplexArray(int iSize, bool iPlanar = true) : size(iSize), planar(iPlanar) {
        data = new AlignedArray<D>(2*size);
        planarGroupSize = planar ? size : 1;
    }

    D* getReals() { return &(*data)[0]; }
    D* getImaginaries() { return planar ? &(*data)[size] : &(*data)[1]; }

    D* getData() { return &(*data)[0]; }


    void setElement(int index, Complex<D> value) {
        if (planar) {
            (*data)[index] = value.getReal();
            (*data)[planarGroupSize + index] = value.getImaginary();
        } else {
            (*data)[2*index] = value.getReal();
            (*data)[2*index + 1] = value.getImaginary();
        }
    }

    Complex<D> getElement(int index, Complex<D> value) {
        Complex<D> result;
        if (planar) { return Complex<D>((*data)[index], (*data)[planarGroupSize + index]); }
        else { return Complex<D>((*data)[2*index], (*data)[2*index + 1]); }
    }

    int getSize() const { return size; }
    bool getPlanar() const { return planar; }

    ~ComplexArray() { delete data; }
};
*/
#endif
