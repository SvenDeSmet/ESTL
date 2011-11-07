#ifndef KERNELGENERATOR_H
#define KERNELGENERATOR_H

// Utility classes for kernel generators

#include <string>
#include <vector>
#include <sstream>
#define _USE_MATH_DEFINES
#include "math.h"

typedef std::string streng;

streng intToStr(int i);

class Expression {
private:
    streng representation;
public:
    Expression() {}
    Expression(streng iRepresentation) : representation(iRepresentation) { }
    virtual streng getRepresentation() { return representation; }
    streng operator () () { return (streng) *this; }
    operator streng () { return getRepresentation(); }
};


/*
class StructureValue : public Value {
private:
public:
    StructureValue(streng iRepresentation) : Value(iRepresentation) { }
    StructureValue() { }

    virtual Value operator [] (int ix) { return Value((*this)() + streng(".") + ); } // Field with index ix
};*/

class KomplexMath {
public:
    static streng getDeclarations() {
        std::stringstream result; result << "\
        typedef float T;\n\
        typedef float2 K;\n\
        inline K komplex(T iR, T iI);       inline K komplex(T iR, T iI) { K k; k.x = iR; k.y = iI; return k; };\n\
        inline K unit(int n, int d);\n\
        inline K mul(const K a, const K b); inline K mul(const K a, const K b) { return komplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); };\n\
        inline K add(const K a, const K b); inline K add(const K a, const K b) { return komplex(a.x + b.x, a.y + b.y); }\n\
        inline K unit(int n, int d) { const float frac_PI2_d = " << (2*M_PI) << "f/d; return komplex(native_cos(frac_PI2_d*n), native_sin(frac_PI2_d*n)); };\n;";
        return result.str();
    }
};

class KomplexUnit : public Expression {
private:
    bool negative;
public:
    int numerator, denominator;

    KomplexUnit(int iNumerator, int iDenominator, bool iNegative = true) : negative(iNegative), numerator(iNumerator), denominator(iDenominator) { }
    streng toFloatString() {
        //double alpha = (negative ? -1 : 1)*2*M_PI*(((double) numerator)/denominator);
        std::stringstream result;
//        result << streng("unit(") << numerator << ", " << denominator << ")";
        result << streng("komplex(") << cos((2*M_PI*numerator)/denominator) << "f, " << sin((2*M_PI*numerator)/denominator) << "f)";
        return result.str();
    }

    virtual streng getRepresentation() { return toFloatString(); }
};

class Assignment : public Expression {
private:
    Expression to, from;
public:
    Assignment(Expression iTo, Expression iFrom) : to(iTo), from(iFrom) { }

    virtual streng getRepresentation() { return to() + streng(" = ") + from() + streng(";"); }
};

class IntegerDivision : public Expression {
private:
    streng numerator;
    int denominator;
    int bits;
public:
    IntegerDivision(streng iNumerator, int iDenominator, int iBits = 24) : numerator(iNumerator), denominator(iDenominator), bits(iBits) { }

    virtual streng getRepresentation() { if (denominator == 1) return numerator;//return numerator + streng("/") + intToStr(denominator);
        switch (bits) {
            case 32: return streng("(((long) (") + numerator + streng("))*") + intToStr((((1 << bits) + (denominator - 1))/denominator)) + streng(") >> ") + intToStr(bits);
            case 16: return streng("(((int) (") + numerator + streng("))*") + intToStr((((1 << bits) + (denominator - 1))/denominator)) + streng(") >> ") + intToStr(bits);
            case 8: return streng("(((short int) (") + numerator + streng("))*") + intToStr((((1 << bits) + (denominator - 1))/denominator)) + streng(") >> ") + intToStr(bits);
            default: return streng("(((long) (") + numerator + streng("))*") + intToStr((((1 << bits) + (denominator - 1))/denominator)) + streng(") >> ") + intToStr(bits);
        }
    }
};

int mod(int a, int b); //{ return (a - b*(a/b)); }

class KomplexConstMultiplication : public Expression {
private:
public:
    KomplexUnit a;
    Expression b;
    KomplexConstMultiplication(KomplexUnit iA, Expression iB) : a(iA), b(iB) { }

    virtual streng getRepresentation() {
            streng bRep = b.getRepresentation();
            int num = mod(a.numerator, a.denominator);
            if (mod(4*num, a.denominator) == 0) { // 1 + 0 i --- 0 + 1 i --- -1 + 0 i --- 0 - 1 i
                if (num == 0) {
                    return bRep;
                } else if ((2*num == a.denominator) || (2*num == -a.denominator)) {
                    return streng("komplex(-") + bRep + streng(".x, -") + bRep + streng(".y)");
                } else if ((4*num == a.denominator) || (4*num == -3*a.denominator)) {
                    return streng("komplex(-") + bRep + streng(".y, ") + bRep + streng(".x)");
                } else {
                    return streng("komplex(") + bRep + streng(".y, -") + bRep + streng(".x)");
                }// else return streng("mul(") + a.getRepresentation() + streng(", ") + b.getRepresentation() + streng(")");
            }/* else if (mod(8*num, a.denominator) == 0) {
                std::stringstream ss;
                ss << cos(2*M_PI/8) << "f";
                streng fos2 = ss.str();
                //(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r)
                if ((8*num == a.denominator) || (8*num == -7*a.denominator)) {
                    return streng("komplex(") + fos2 + streng("*(") + bRep + streng(".r - ") + bRep + streng(".i), ")
                                              + fos2 + streng("*(") + bRep + streng(".i + ") + bRep + streng(".r))");
                } else if ((8*num == -a.denominator) || (8*num == 7*a.denominator)) {
                    return streng("komplex(") + fos2 + streng("*(") + bRep + streng(".r + ") + bRep + streng(".i), ")
                                              + fos2 + streng("*(") + bRep + streng(".i - ") + bRep + streng(".r))");
                } else if ((8*num == 3*a.denominator) || (8*num == -5*a.denominator)) {
                    return streng("komplex(-") + fos2 + streng("*(") + bRep + streng(".r + ") + bRep + streng(".i), ")
                                              + fos2 + streng("*(") + bRep + streng(".r - ") + bRep + streng(".i))");
                } else {
                    return streng("komplex(") + fos2 + streng("*(") + bRep + streng(".i - ") + bRep + streng(".r), -")
                                              + fos2 + streng("*(") + bRep + streng(".i + ") + bRep + streng(".r))");
                }// else return streng("mul(") + a.getRepresentation() + streng(", ") + b.getRepresentation() + streng(")");
            }*/ else {
                return streng("mul(") + a.getRepresentation() + streng(", ") + b.getRepresentation() + streng(")");
            }
    }
};

class Array {
private:
    streng type;
    int length;
    bool inRegisters, structArray;
    streng name;
    std::vector<streng> items;
public:
    Array(streng iName) : name(iName) { items.push_back("x"); items.push_back("y"); structArray = true; inRegisters = false; }
    Array(streng iType, int iLength, bool iInRegisters, streng iName) : type(iType), length(iLength), inRegisters(iInRegisters), name(iName), structArray(false) { }

    streng getDeclaration() {
        streng result;
        if (inRegisters) { result += type + streng(" ");
            for (int l = 0; l < length; ++l) {
                if (l > 0) result += ", ";
                result += name + "_" + intToStr(l);
            }
        } else { result = type + streng(" ") + name + streng("[") + intToStr(length) + streng("]"); }

        return result + streng(";");
    }

    virtual Expression getItem(int ix) {
        if (inRegisters) { return name + "_" + intToStr(ix); }
        else if (structArray) { return name + streng(".") + items[ix]; }
        else { return name + streng("[") + intToStr(ix) + streng("]"); }
    }

    virtual Expression getItem(Expression ix) {
        if (inRegisters) { /* throw exception*/ }
        else { return name + streng("[") + ix() + streng("]"); }
    }

    Expression operator [] (int ix) { return getItem(ix); }
    Expression operator [] (Expression ix) { return getItem(ix); }
    Expression operator [] (streng ix) { return (*this)[Expression(ix)]; }

//    virtual Expression assignToItem(int ix, Expression fromV)        { return getItem(ix)() + streng(" = ") + fromV() + streng(";"); }
  //  virtual Expression assignToItem(Expression ix, Expression fromV) { return getItem(ix)() + streng(" = ") + fromV() + streng(";"); }

    //virtual Expression assignFromItem(int ix, Expression toV)        { return toV() + streng(" = ") + getItem(ix)() + streng(";"); }
    //virtual Expression assignFromItem(Expression ix, Expression toV) { return toV() + streng(" = ") + getItem(ix)() + streng(";"); }
};

template <class A>
class CLArrayPlannarizer {
private:
    A& array;
    int plannarLevel, elementLevel, plannarMask;
public:
    CLArrayPlannarizer(A& iArray, int iPlannarLevel, int iElementLevel) : array(iArray), plannarLevel(iPlannarLevel), elementLevel(iElementLevel), plannarMask((1 << plannarLevel) - 1) { }

    Expression getPosition(Expression index) const {
        return streng("(((") + index() + streng(") >> ") + intToStr(plannarLevel) + streng(") << ") + intToStr(plannarLevel + elementLevel) + streng(") | ((")
         + index() + streng(") & ") + intToStr(plannarMask) + streng(")");
    }

    Expression getElementPosition(Expression index, Expression element) const {
        return index() + streng(" | (") + element() + streng(" << ") + intToStr(plannarLevel) + streng(")");
    }

    Expression getElementPosition(Expression index, int element) const {
        return index() + ((element > 0) ? (streng(" | (") + intToStr(element) + streng(" << ") + intToStr(plannarLevel) + streng(")")) : streng(""));
    }

    virtual Expression assignToItem(Expression ix, Expression element, Expression fromV) { std::stringstream result; result << "{";
        result << "int index = " << getElementPosition(getPosition(ix), element)() << ";";
        result << array["index"]() << " = " << fromV() << ";";
        result << "}";
        return result.str();
    }

    virtual Expression assignFromItem(Expression ix, Expression element, Expression toV) { std::stringstream result; result << "{";
        result << "int index = " << getElementPosition(getPosition(ix), element)() << ";";
        result << toV()  << " = " << array["index"]()<< ";";
        result << "}";
        return result.str();
    }

    virtual Expression assignToItem(Expression ix, Array toV) { std::stringstream result; result << "{";
        result << "int index = " << getPosition(ix)() << ";";
        for (int e = 0; e < (1 << elementLevel); ++e) {
            result << toV[e]() << " = " << array[getElementPosition(Expression("index"), e)]() << ";";
        }
        result << "}";
        return result.str();
    }

    virtual Expression assignFromItem(Expression ix, Array fromV) { std::stringstream result; result << "{";
        result << "int index = " << getPosition(ix)() << ";";
        for (int e = 0; e < (1 << elementLevel); ++e) {
            result << array[getElementPosition(Expression("index"), e)]() << " = " << fromV[e]() << ";";
        }
        result << "}";
        return result.str();
    }
};

class PlannarizedComplexCLArray {
private:
    const int plannarLevel;
    CLArrayPlannarizer<Array>* planarizedData;
    Array* data;
    int size, elements;
    streng name;
public:
    PlannarizedComplexCLArray(int iPlannarLevel, streng iName, int iSize = -1) : plannarLevel(iPlannarLevel), size(iSize), name(iName) {
        int roundLevel = plannarLevel + 1;
        elements = 2*size;
        elements = roundLevel*((elements + ((1 << roundLevel) - 1))/roundLevel);
        data = new Array("float", size, false, name);
        planarizedData = new CLArrayPlannarizer<Array>(*data, plannarLevel, 1);
    }

    int getElements() const { return elements; }
    int getSize() const { return size; }

    Expression assignToItem(Expression index, Array v) { return planarizedData->assignToItem(index, v); }
    Expression assignFromItem(Expression index, Array v) { return planarizedData->assignFromItem(index, v); }

    ~PlannarizedComplexCLArray() { delete planarizedData; delete data; }
};

#endif
