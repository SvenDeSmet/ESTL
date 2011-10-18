/* The information in this file is
 * Copyright (C) 2011, Sven De Smet <sven@cubiccarrot.com>
 * and is subject to the terms and conditions of the
 * GNU Lesser General Public License Version 2.1
 * The license text is available from
 * http://www.gnu.org/licenses/lgpl.html
 */

#include "ContiguousKernelGenerator.h"
#include "stdio.h"

std::string intToStr(int i) { if (i == 0) return "0";
    char* p = new char[(int) (ceil(log((double) i)/log((double) 10)) + 2)];
    sprintf(p, "%i", i);
    std::string result = p;
    delete [] p;
    return result;
}

int mod(int a, int b) { return (a - b*(a/b)); }

