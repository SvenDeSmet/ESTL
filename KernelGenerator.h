#ifndef KERNELGENERATOR_H
#define KERNELGENERATOR_H

#include <string>

typedef std::string streng;

std::string intToStr(int i);

class Expression {
public:
    virtual streng getRepresentation() = 0;
};

class Value : public Expression {
private:
    streng representation;
public:
    Value(streng iRepresentation) : representation(iRepresentation) { }
    Value() { }

    virtual streng getRepresentation() { return representation; }
    streng operator()() { return getRepresentation(); }
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

int mod(int a, int b); //{ return (a - b*(a/b)); }

class KomplexConstMultiplication : public Expression {
private:
public:
    KomplexUnit a;
    Value b;
    KomplexConstMultiplication(KomplexUnit iA, Value iB) : a(iA), b(iB) { }

    virtual streng getRepresentation() {
            streng bRep = b.getRepresentation();
            int num = mod(a.numerator, a.denominator);
            if (mod(4*num, a.denominator) == 0) { // 1 + 0 i --- 0 + 1 i --- -1 + 0 i --- 0 - 1 i
                if (num == 0) {
                    return bRep;
                } else if ((2*num == a.denominator) || (2*num == -a.denominator)) {
                    return streng("komplex(-") + bRep + streng(".r, -") + bRep + streng(".i)");
                } else if ((4*num == a.denominator) || (4*num == -3*a.denominator)) {
                    return streng("komplex(-") + bRep + streng(".i, ") + bRep + streng(".r)");
                } else {
                    return streng("komplex(") + bRep + streng(".i, -") + bRep + streng(".r)");
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
    bool inRegisters;
    streng name;
public:
    Array(streng iType, int iLength, bool iInRegisters, streng iName) : type(iType), length(iLength), inRegisters(iInRegisters), name(iName) { }

    streng getDeclaration() {
        streng result;
        if (inRegisters) { result += type + streng(" ");
            for (int l = 0; l < length; ++l) {
                if (l > 0) result += ", ";
                result += name + intToStr(l);
            }
        } else {
            result = type + streng(" ") + name + streng("[") + intToStr(length) + streng("]");
        }
        return result + streng(";");
    }

    Value getItem(int ix) {
        if (inRegisters) { return name + intToStr(ix); }
        else { return name + streng("[") + intToStr(ix) + streng("]"); }
    }
};

#endif // KERNELGENERATOR_H
