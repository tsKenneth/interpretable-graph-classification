#ifndef cfg_H
#define cfg_H

#include <iostream>
#include <cstring>
#include <fstream>
#include <set>
#include <map>

typedef float Dtype;

struct cfg
{
    static bool msg_average;

    static void LoadParams(const int argc, const char** argv)
    {
        for (int i = 1; i < argc; i += 2)
        {
            if (strcmp(argv[i], "-msg_average") == 0)
                msg_average = atoi(argv[i + 1]);
        }
    }
};

#endif
