/* The information in this file is
 * Copyright (C) 2011, Sven De Smet <sven@cubiccarrot.com>
 * and is subject to the terms and conditions of the
 * GNU Lesser General Public License Version 2.1
 * The license text is available from
 * http://www.gnu.org/licenses/lgpl.html
 */

#include "Complex.h"

template <>
Complex<short int> Complex<short int>::unit(double iPhase) { return Complex<short int>((short int) (16384*cos(iPhase)), (short int) (16384*sin(iPhase))); }

template <> Complex<short int> inline operator >> (const Complex<short int> a, const int shift) { return Complex<short int>(a.r >> shift, a.i >> shift); }
template <> Complex<short int> inline operator + (const Complex<short int> a, const Complex<short int> b) { return Complex<short int>(iadd16(a.r, b.r), iadd16(a.i, b.i)); }
template <> Complex<short int> inline operator - (const Complex<short int> a, const Complex<short int> b) {	return Complex<short int>(isub16(a.r, b.r), isub16(a.i, b.i)); }
template <> short int Complex<short int>::getNorm() const { return 32768*sqrt(getNormSquared()/32768.0); }
