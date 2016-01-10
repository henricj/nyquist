//
//  internal.c
//  
//
//
//

#define _USE_MATH_DEFINES
#include "internal.h"

float hann(double x)
{
    return 0.5 * (1.0 - cos(2 * M_PI * x));
}

float hamm(double x)
{
    return 0.54 - 0.46 * cos(2 * M_PI * x);
}

void OneDimensionFFTshift(float vector[], int VectorLength)
{
    float temp;
    for (int i = 0; i <= VectorLength / 2 - 1; i++)
    {
        temp = vector[i];
        vector[i] = vector[VectorLength / 2 + i];
        vector[VectorLength / 2 + i] = temp;
    }
}
