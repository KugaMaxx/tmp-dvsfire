#ifndef EDT_H
#define EDT_H

#include <stdlib.h>
#include <iostream>

namespace edt {

class EventDetector {
public:
    int16_t sizeX;
    int16_t sizeY;
    size_t  _LENGTH_;  // sizeX * sizeY
};

}

#endif
