#ifndef BASE64_H
#define BASE64_H

#include <Arduino.h>

namespace base64 {
    String encode(const uint8_t* data, size_t length);
}

#endif