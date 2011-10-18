#ifndef SIMPLEMATH_H
#define SIMPLEMATH_H

static inline int mIn(int a, int b) { return (a < b) ? a : b; }
static inline int mAx(int a, int b) { return (a > b) ? a : b; }

static inline int ceilog(int n, int R) {
    int result = 0;
    for (int num = 1; num < n; num *= R) result++;
    return result;
}

static inline int ceilint(int n, int R) { return R*((n + (R - 1))/R); }

#endif // SIMPLEMATH_H
