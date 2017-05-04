#include <iostream>

float eps_float()
{
    float epsilon = 1;
    while ((float)(1.0 + epsilon) > 1.0)
    {
        epsilon = (float)(epsilon*0.5);
    }
    return epsilon;
}

double eps_double()
{
    double epsilon = 1.0;
    while ((double)(1.0 + epsilon) > 1.0)
    {
        epsilon = (double)(epsilon*0.5);
    }
    return epsilon;
}

long double eps_ldouble():
{
    long double epsilon = 1.0;
    while ((long double)(1.0 + epsilon) > 1.0)
    {
        epsilon = (long double)(epsilon*0.5);
    }
    return epsilon
}

int main()
{
    float eps1;
    double eps2;
    long double eps3;
    eps1 = eps_float();
    eps2 = eps_double();
    eps3 = eps_ldouble();
    std::cout << "Machine epsilon for float = " << eps1 << std::endl;
    std::cout << "Machine epsilon for double = " << eps2 << std::endl;
    std::cout << "Machine epsilon for long double = " << eps3 << std::endl;
    return 0;
}