#include "base64.h"

static const char PROGMEM b64_alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                           "abcdefghijklmnopqrstuvwxyz"
                                           "0123456789+/";

String base64::encode(const uint8_t* data, size_t length) {
    String encoded = "";
    size_t i = 0;
    
    while (i < length) {
        uint32_t octet_a = i < length ? data[i++] : 0;
        uint32_t octet_b = i < length ? data[i++] : 0;
        uint32_t octet_c = i < length ? data[i++] : 0;

        uint32_t triple = (octet_a << 0x10) + (octet_b << 0x08) + octet_c;

        encoded += (char)pgm_read_byte(&b64_alphabet[(triple >> 3 * 6) & 0x3F]);
        encoded += (char)pgm_read_byte(&b64_alphabet[(triple >> 2 * 6) & 0x3F]);
        encoded += (char)pgm_read_byte(&b64_alphabet[(triple >> 1 * 6) & 0x3F]);
        encoded += (char)pgm_read_byte(&b64_alphabet[(triple >> 0 * 6) & 0x3F]);
    }

    // Adiciona padding se necessário
    switch (length % 3) {
        case 1:
            encoded.setCharAt(encoded.length() - 1, '=');
            encoded.setCharAt(encoded.length() - 2, '=');
            break;
        case 2:
            encoded.setCharAt(encoded.length() - 1, '=');
            break;
    }

    return encoded;
}